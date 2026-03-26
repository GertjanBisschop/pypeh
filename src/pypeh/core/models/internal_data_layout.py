from __future__ import annotations

import itertools
import logging

from collections import defaultdict
from dataclasses import dataclass, field
from peh_model import peh
from typing import TYPE_CHECKING, Generator, Generic, Sequence, Protocol, Type
from ulid import ULID

from pypeh.core.cache.containers import CacheContainer, CacheContainerView
from pypeh.core.models.typing import T_DataType
from pypeh.core.models.constants import ObservablePropertyValueType

if TYPE_CHECKING:
    from typing import Any
    from pypeh.core.interfaces.outbound.dataops import OutDataOpsInterface


logger = logging.getLogger(__name__)


class ContextIndexProtocol(Protocol):
    def context_lookup(self, observation_id: str, observable_property_id: str) -> tuple[str, str]: ...


@dataclass
class JoinSpec:
    left_element: str
    left_dataset: str
    right_element: str
    right_dataset: str


@dataclass
class DatasetSchemaElement:
    label: str
    observable_property_id: str
    data_type: ObservablePropertyValueType
    identifier: str = field(default_factory=lambda: str(ULID()))


@dataclass
class ElementReference:
    dataset_label: str
    element_label: str


@dataclass
class ForeignKey:
    element_label: str
    reference: ElementReference
    identifier: str = field(default_factory=lambda: str(ULID()))


@dataclass
class DatasetSchema:
    elements: dict[str, DatasetSchemaElement] = field(default_factory=dict)
    primary_keys: set[str] = field(default_factory=set)
    foreign_keys: dict[str, ForeignKey] = field(default_factory=dict)
    identifier: str = field(default_factory=lambda: str(ULID()))
    factory: DatasetSeriesBuilder | None = field(default=None)

    def __post_init__(self):
        self._type = self.get_type_annotations()
        self._elements_by_observable_property: dict[str, set[str]] = self.build_observable_property_index()
        if self.factory is None:
            self.factory = DatasetSeriesBuilder()

    def get_type_annotations(self) -> dict[str, ObservablePropertyValueType]:
        ret: dict[str, ObservablePropertyValueType] = dict()
        for element in self.elements.values():
            data_type = element.data_type
            if data_type is not None:
                ret[element.label] = data_type

        return ret

    def build_observable_property_index(self) -> dict[str, set[str]]:
        # TODO: fix this, a single dataset can contain multiple occurrences of the same observable property
        elements_by_observable_property = defaultdict(set)
        for element_label, element in self.elements.items():
            elements_by_observable_property[element.observable_property_id].add(element_label)
        return elements_by_observable_property

    def get_type(self, element_label: str) -> ObservablePropertyValueType:
        return self._type[element_label]

    def get_element_by_label(self, element_label: str) -> DatasetSchemaElement | None:
        return self.elements.get(element_label, None)

    def yield_elements_by_observable_property_id(
        self, observable_property_id: str
    ) -> Generator[DatasetSchemaElement, None, None]:
        element_labels = self._elements_by_observable_property[observable_property_id]
        for element in element_labels:
            ret = self.get_element_by_label(element_label=element)
            assert ret is not None
            yield ret

    def get_observable_property_ids(self) -> list[str]:
        return [element.observable_property_id for element in self.elements.values()]

    def get_element_labels(self) -> list[str]:
        return list(self.elements.keys())

    def add_observable_property(
        self,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        element_label: str | None = None,
        is_primary_key: bool = False,
    ) -> DatasetSchemaElement:
        assert self.factory is not None
        if element_label is None:
            element_label = observable_property_id
        if element_label in self.elements:
            raise ValueError(f"DatasetSchema already contains an Element with label {element_label}")
        assert isinstance(data_type, ObservablePropertyValueType)
        new_element = self.factory.mint_dataset_schema_element(
            label=element_label,
            observable_property_id=observable_property_id,
            data_type=data_type,
        )
        self.elements[element_label] = new_element
        self._type[element_label] = data_type
        if observable_property_id in self._elements_by_observable_property:
            raise ValueError(f"ObservableProperty with id {observable_property_id} already in schema")
        self._elements_by_observable_property[observable_property_id].add(element_label)
        if is_primary_key:
            self.primary_keys.add(element_label)

        return new_element

    def add_foreign_key_link(
        self,
        element_label: str,
        foreign_key_dataset_label: str,
        foreign_key_element_label: str,
    ):
        assert self.factory is not None
        foreign_key_object = self.factory.mint_foreign_key(
            element_label=element_label,
            reference=ElementReference(
                dataset_label=foreign_key_dataset_label,
                element_label=foreign_key_element_label,
            ),
        )
        self.foreign_keys[element_label] = foreign_key_object

    # TODO: move method, this is probably not the right location
    def apply_context_to_expression(
        self,
        expression: peh.ValidationExpression,
        context: dict[str, dict[str, str]],
        this_dataset: str,  # temporary fix
    ):
        expression_stack = [expression]
        while expression_stack:
            expression = expression_stack.pop()
            assert isinstance(expression, peh.ValidationExpression)
            conditional_expression = expression.validation_condition_expression
            if conditional_expression is not None:
                assert isinstance(conditional_expression, peh.ValidationExpression)
                expression_stack.append(conditional_expression)
            arg_expressions = expression.validation_arg_expressions
            if arg_expressions is not None:
                for arg_expression in arg_expressions:
                    assert isinstance(arg_expression, peh.ValidationExpression)
                    expression_stack.append(arg_expression)

            # apply context
            arg_field_references = expression.validation_arg_contextual_field_references
            if arg_field_references is not None:
                for arg_ref in arg_field_references:
                    assert isinstance(arg_ref, peh.ContextualFieldReference)
                    field_label = arg_ref.field_label
                    if field_label in context:
                        retrieved_context = context[field_label]
                        if len(retrieved_context) == 1:
                            new_dataset_label = next(iter(retrieved_context))
                        else:
                            new_dataset_label = this_dataset
                        new_field_label = retrieved_context[new_dataset_label]
                        arg_ref.dataset_label = new_dataset_label
                        arg_ref.field_label = new_field_label

            subject_field_references = expression.validation_subject_contextual_field_references
            if subject_field_references is not None:
                for subject_field_ref in subject_field_references:
                    assert isinstance(subject_field_ref, peh.ContextualFieldReference)
                    field_label = subject_field_ref.field_label
                    if field_label in context:
                        retrieved_context = context[field_label]
                        if len(retrieved_context) == 1:
                            new_dataset_label = next(iter(retrieved_context))
                        else:
                            new_dataset_label = this_dataset
                        new_field_label = retrieved_context[new_dataset_label]
                        subject_field_ref.dataset_label = new_dataset_label
                        subject_field_ref.field_label = new_field_label

    def apply_context(
        self,
        context: dict[str, dict[str, str]],
        cache: CacheContainer,
        this_dataset: str,  # temporary fix
    ):
        """
        ObservableProperties require context for:
        - CalculationDesigns
        - ValidationDesigns
        """
        for schema_element in self.elements.values():
            observable_property_id = schema_element.observable_property_id
            observable_property = cache.get(observable_property_id, "ObservableProperty")
            assert isinstance(observable_property, peh.ObservableProperty)
            validation_designs = observable_property.validation_designs
            if validation_designs is not None:
                for validation_design in validation_designs:
                    assert isinstance(validation_design, peh.ValidationDesign)
                    expression = validation_design.validation_expression
                    assert isinstance(expression, peh.ValidationExpression)
                    self.apply_context_to_expression(expression, context, this_dataset=this_dataset)

            # CALCULATION DESIGN EXCLUDED FOR NOW
            # if observable_property.calculation_design is not None:
            #    pass

    #### RESTRUCTURE DATASETSCHEMA ####

    def subset(self, element_group: Sequence[str]) -> DatasetSchema:
        elements = {}
        foreign_keys = {}
        primary_keys = set()
        assert self.factory is not None

        for element_label in element_group:
            element = self.get_element_by_label(element_label)
            assert element is not None, f"Element with label {element_label} not found in schema"
            elements[element_label] = element
            if element_label in self.foreign_keys:
                foreign_keys[element_label] = self.foreign_keys[element_label]
            if self.primary_keys is not None:
                if element_label in self.primary_keys:
                    primary_keys.add(element_label)
            elements[element_label] = element

        ret = self.factory.mint_dataset_schema(
            elements=elements,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
        )
        return ret

    def relabel(self, element_mapping: dict[str, str]):
        elements: dict[str, DatasetSchemaElement] = dict()
        all_type_info: dict[str, ObservablePropertyValueType] = dict()
        elements_by_observable_property: dict[str, set[str]] = defaultdict(set)
        primary_keys: set[str] | None = set()
        foreign_keys: dict[str, ForeignKey] = {}

        for element_label, new_element_label in element_mapping.items():
            schema_element = self.elements.pop(element_label)
            elements[new_element_label] = schema_element
            schema_element.label = new_element_label

            # _type dict
            type_info = self._type.pop(element_label)
            all_type_info[new_element_label] = type_info

            # _elements_by_observable_property
            observable_property_id = schema_element.observable_property_id
            element_labels = self._elements_by_observable_property.pop(observable_property_id)
            for element_label in element_labels:
                elements_by_observable_property[observable_property_id].add(element_label)

            # primary_keys
            if self.primary_keys is not None:
                if element_label in self.primary_keys:
                    self.primary_keys.discard(element_label)
                    primary_keys.add(new_element_label)

            # foreign_keys
            if element_label in self.foreign_keys:
                foreign_key = self.foreign_keys.pop(element_label)
                foreign_keys[new_element_label] = foreign_key

        if len(self.elements) > 0:
            for element in self.elements:
                if element in elements:
                    raise ValueError("Schema element label {element} is non unique")
            elements = {**elements, **self.elements}
            all_type_info = {**all_type_info, **self._type}
            elements_by_observable_property = {
                **elements_by_observable_property,
                **self._elements_by_observable_property,
            }
            foreign_keys = {**foreign_keys, **self.foreign_keys}
            if self.primary_keys is not None:
                primary_keys = primary_keys | self.primary_keys
            else:
                assert len(primary_keys) == 0
                primary_keys = None

        num_elements = len(elements)
        assert len(elements_by_observable_property) == num_elements
        assert len(all_type_info) == num_elements
        assert len(foreign_keys) <= num_elements

        self.elements = elements
        self._elements_by_observable_property = elements_by_observable_property
        self._type = all_type_info
        self.foreign_keys = foreign_keys
        if primary_keys is not None:
            self.primary_keys = primary_keys

    def __len__(self):
        return len(self.elements)

    #### EXTRACT INFO FROM SCHEMA ####

    def detect_join(
        self,
        dataset_label: str,
        other_schema: DatasetSchema,
        other_dataset_label: str,
    ) -> list[JoinSpec] | None:
        # Case 1: A → B directly
        for col, fk in self.foreign_keys.items():
            if fk.reference.dataset_label == other_dataset_label:
                return [
                    JoinSpec(
                        left_element=fk.element_label,
                        left_dataset=dataset_label,
                        right_element=fk.reference.element_label,
                        right_dataset=other_dataset_label,
                    )
                ]

        # Case 2: B → A directly
        for col, fk in other_schema.foreign_keys.items():
            if fk.reference.dataset_label == dataset_label:
                return [
                    JoinSpec(
                        left_element=fk.reference.element_label,
                        left_dataset=dataset_label,
                        right_element=fk.element_label,
                        right_dataset=other_dataset_label,
                    )
                ]

        # Case 3: shared third dataset: requires two `JoinSpec`
        refs_a = {
            fk.reference.dataset_label: (fk.element_label, fk.reference.element_label)
            for fk in self.foreign_keys.values()
        }
        refs_b = {
            fk.reference.dataset_label: (fk.element_label, fk.reference.element_label)
            for fk in other_schema.foreign_keys.values()
        }

        shared = set(refs_a.keys()).intersection(set(refs_b.keys()))
        if shared:
            shared_label = next(iter(shared))
            a_col_local, a_other = refs_a[shared_label]
            b_col_local, b_other = refs_b[shared_label]

            return [
                JoinSpec(
                    left_element=a_col_local,
                    left_dataset=dataset_label,
                    right_element=a_other,
                    right_dataset=shared_label,
                ),
                JoinSpec(left_element=b_col_local, left_dataset="", right_element=b_other, right_dataset=shared_label),
            ]

        return None


@dataclass(kw_only=True)
class Resource:
    """
    Parent class for Dataset and DatasetSeries
    """

    label: str
    identifier: str = field(default_factory=lambda: str(ULID()))
    metadata: dict[str, Any] = field(default_factory=dict)

    factory: DatasetSeriesBuilder | None = field(default=None)

    def __post_init__(self):
        if self.factory is None:
            self.factory = DatasetSeriesBuilder()

    def add_metadata(self, metadata_key: str, metadata_value: Any) -> bool:
        if metadata_key in self.metadata:
            raise KeyError(f"{metadata_key} key already used in metadata mapping")
        self.metadata[metadata_key] = metadata_value

        return True

    @property
    def described_by(self) -> str | peh.NamedThingId | None:
        return self.metadata.get("described_by", None)


@dataclass(kw_only=True)
class Dataset(Resource, Generic[T_DataType]):
    schema: DatasetSchema | None = field(default=None)
    data: T_DataType | None = field(default=None)
    part_of: DatasetSeries | None = field(default=None)
    observation_ids: set[str] = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()
        if self.schema is None:
            assert self.factory is not None
            self.schema = self.factory.mint_dataset_schema()

    def get_type_annotations(self) -> dict[str, ObservablePropertyValueType]:
        return self.schema.get_type_annotations()

    #### EXTRACT INFO FROM DATASET ####

    def get_element_labels(self) -> list[str]:
        return self.schema.get_element_labels()

    def get_schema_element_by_label(self, element_label: str) -> DatasetSchemaElement | None:
        return self.schema.get_element_by_label(element_label)

    def get_schema_element_by_observable_property_id(
        self, observable_property_id: str
    ) -> Generator[DatasetSchemaElement, None, None]:
        yield from self.schema.yield_elements_by_observable_property_id(observable_property_id)

    def get_observable_property_ids(self) -> list[str]:
        return self.schema.get_observable_property_ids()

    def get_primary_keys(self) -> set[str] | None:
        return self.schema.primary_keys

    def resolve_join(self, other: Dataset) -> list[JoinSpec] | None:
        schema = self.schema
        assert schema is not None
        return schema.detect_join(dataset_label=self.label, other_schema=other.schema, other_dataset_label=other.label)

    #### RESTRUCTURE DATASET ####

    def subset(
        self,
        new_dataset_series_label: str,
        observation_groups: dict[str, list[peh.Observation]],
        observation_designs: list[peh.ObservationDesign],
        dataops_adapter: OutDataOpsInterface,
    ) -> DatasetSeries:
        # TODO: schemas for all new datasets need to be properly relabeled
        assert self.factory is not None
        ret = self.factory.mint_dataset_series(label=new_dataset_series_label, parts={})
        observation_design_dict = {od.id: od for od in observation_designs}

        for dataset_label, observation_group in observation_groups.items():
            observable_property_ids: set[str] = set()
            selected_observation_ids: set[str] = set()
            for observation in observation_group:
                assert observation.id in self.observation_ids
                selected_observation_ids.add(observation.id)
                observation_design_id = str(observation.observation_design)
                assert observation_design_id in observation_design_dict.keys()
                observation_design = observation_design_dict[observation_design_id]
                assert observation_design is not None
                assert isinstance(observation_design, peh.ObservationDesign)
                for observable_property_specification in observation_design.observable_property_specifications:
                    assert observable_property_specification is not None
                    assert observable_property_specification.observable_property is not None
                    observable_property_ids.add(str(observable_property_specification.observable_property))

            element_group = []
            for observable_property_id in observable_property_ids:
                elements = self.schema.yield_elements_by_observable_property_id(observable_property_id)
                for element in elements:
                    element_group.append(element.label)
            assert len(element_group) > 0

            # split data
            data_subset = dataops_adapter.subset(
                data=self.data,
                element_group=element_group,
            )
            # split schema
            schema_subset = self.schema.subset(element_group)
            # add both to new dataset
            new_dataset = self.factory.mint_dataset(
                schema=schema_subset,
                label=dataset_label,
                data=data_subset,
                observation_ids=selected_observation_ids,
                part_of=ret,
            )
            ret.parts[dataset_label] = new_dataset

        return ret

    def relabel(self, element_mapping: dict[str, str], dataops_adapter: OutDataOpsInterface) -> bool:
        # uniqueness check
        if len(set(element_mapping.values())) != len(element_mapping):
            raise ValueError("Not all values in the element_mapping are unique")
        # relabel schema
        _ = self.schema.relabel(element_mapping)
        # relabel dataset
        self.data = dataops_adapter.relabel(self.data, element_mapping)

        return True

    def add_observable_property(
        self,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        element_label: str | None = None,
        is_primary_key: bool = False,
    ) -> DatasetSchemaElement:
        return self.schema.add_observable_property(
            observable_property_id,
            data_type,
            element_label,
            is_primary_key,
        )

    def add_observation_to_index(self, observation_id: str):
        if self.part_of:
            self.part_of._register_observation(observation_id, self.label)
        self.observation_ids.add(observation_id)

    def remove_observation_from_index(self, observation_id: str):
        self.observation_ids.remove(observation_id)
        if self.part_of:
            self.part_of._unregister_observation(observation_id)

    def add_data(
        self,
        data: T_DataType,
        data_labels: list[str] | None = None,
        overwrite: bool = True,
        allow_incomplete: bool = False,
    ):
        if not overwrite:
            if self.data is not None:
                raise NotImplementedError()

        if data_labels is None:
            data_labels = self.get_element_labels()

        if len(self.schema) > 0:
            assert data_labels is not None

            if allow_incomplete:
                assert self.contained_in_schema(data_labels)
            else:
                assert self.matches_schema(data_labels)

            self.data = data

    #### VERIFY DATASET ####

    def contained_in_schema(self, element_labels: list[str]) -> bool:
        """
        Check if the columns from a set of data can be found amongst the labels defined in the dataset schema
        """
        raw_data_labels = set(element_labels)
        schema_labels = set(self.get_element_labels())
        label_diff = raw_data_labels.difference(schema_labels)
        if len(label_diff) != 0:
            raise AssertionError(f"Data Schema Error: label(s) {', '.join(label_diff)} not in schema")
        return len(label_diff) == 0

    def matches_schema(self, element_labels: list[str]) -> bool:
        """
        Check if the columns from a set of data match the labels defined in the dataset schema
        """
        raw_data_labels = set(element_labels)
        schema_labels = set(self.get_element_labels())
        error_str = []
        undefined_diff = raw_data_labels.difference(schema_labels)
        if len(undefined_diff) > 0:
            error_str.append(f"label(s) {', '.join(undefined_diff)} are undefined")
        missing_diff = schema_labels.difference(raw_data_labels)
        if len(missing_diff) > 0:
            error_str.append(f"labels {', '.join(missing_diff)} are missing")
        assert raw_data_labels == schema_labels, f"Data Schema Error: {', '.join(error_str)}"
        return True


@dataclass(kw_only=True)
class DatasetSeries(Resource, Generic[T_DataType]):
    parts: dict[str, Dataset[T_DataType]] = field(default_factory=dict)
    _obs_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _context_index: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )  # {(observation_id, observable_property_id): (dataset_label, dataset_element_label}}

    def _register_observation(
        self,
        observation_id: str,
        dataset_label: str,
    ):
        if observation_id in self._obs_index:
            found_set = self._obs_index[observation_id]
            if dataset_label in found_set:
                raise ValueError(
                    f"Observation with id {observation_id} is already registered for dataset with label {dataset_label}."
                )
        assert (
            dataset_label in self.parts
        ), f"Failed to register observation with id {observation_id}. Could not find dataset with label {dataset_label}"
        self._obs_index[observation_id].add(dataset_label)

    def _unregister_observation(self, observation_id: str, observable_property_ids: list[str] | None = None):
        self._obs_index.pop(observation_id, None)
        if observable_property_ids is not None:
            for obs_prop_id in observable_property_ids:
                self._context_index.pop((observation_id, obs_prop_id))

    def _register_observable_property(
        self,
        observable_property_id: str,
        observation_id: str,
        dataset_label: str,
        element_label: str,
    ):
        self._context_index[(observation_id, observable_property_id)] = (dataset_label, element_label)

    def _unregister_observable_property(
        self,
        observable_property_id: str,
        observation_id: str,
    ):
        self._context_index.pop((observation_id, observable_property_id))

    def build_observation_index(self):
        for dataset_label in self.parts:
            dataset = self[dataset_label]
            assert dataset is not None
            for obs in dataset.observation_ids:
                self._register_observation(obs, dataset_label)

    def build_context_index(self):
        """
        Only works if each dataset only contains one observation
        """
        for dataset_label in self.parts:
            dataset = self[dataset_label]
            assert dataset is not None
            assert (
                len(dataset.observation_ids) == 1
            ), "Cannot build context index if any dataset contains more than one observation"
            observation_id = next(iter(dataset.observation_ids))
            dataset_schema = dataset.schema
            assert dataset_schema is not None
            for element_label, element in dataset.schema.elements.items():
                self._register_observable_property(
                    element.observable_property_id,
                    observation_id,
                    dataset_label,
                    element_label,
                )

    def build_indices(self):
        """
        Building these indices only works if each dataset only contains one observation
        """
        self.build_observation_index()
        self.build_context_index()

    #### CONSTRUCT DATASETSERIES ####
    def apply_context(self, cache: CacheContainer):
        # THIS METHOD MODIFIES THE CACHE !!!
        context = self._get_validation_index()
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            dataset.schema.apply_context(context, cache, this_dataset=self.label)

    def register_dataset(self, dataset: Dataset):
        dataset.part_of = self
        self.parts[dataset.label] = dataset

    def add_data(
        self,
        dataset_label: str,
        data: T_DataType,
        data_labels: list[str] | None = None,
        overwrite: bool = True,
        allow_incomplete: bool = False,
    ):
        dataset = self.parts.get(dataset_label, None)
        assert dataset is not None
        dataset.add_data(
            data=data,
            data_labels=data_labels,
            overwrite=overwrite,
            allow_incomplete=allow_incomplete,
        )

    def add_observable_property(
        self,
        observation_id: str,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        dataset_label: str,
        element_label: str,
        is_primary_key: bool = False,
    ):
        assert dataset_label in self.parts
        dataset = self[dataset_label]
        assert dataset is not None
        self._register_observable_property(
            observable_property_id=observable_property_id,
            observation_id=observation_id,
            dataset_label=dataset_label,
            element_label=element_label,
        )

        return dataset.add_observable_property(
            observable_property_id,
            data_type,
            element_label,
            is_primary_key,
        )

    def add_empty_dataset(self, dataset_label: str, metadata: dict | None = None) -> Dataset:
        assert self.factory is not None
        dataset = self.factory.mint_dataset(
            label=dataset_label,
            factory=self.factory,
        )
        if metadata is not None:
            dataset.metadata = metadata
        self.register_dataset(dataset)

        return dataset

    def add_observation(
        self,
        dataset_label: str,
        observation: peh.Observation,
        labeled_observable_property_specifications: dict[str, peh.ObservablePropertySpecification],
        data: T_DataType | None = None,
    ) -> Dataset:
        """
        labeled_observable_property_specifications: dict[element_label: ObservablePropertySpecification]
        """
        if dataset_label not in self.parts:
            dataset = self.add_empty_dataset(dataset_label)
        else:
            dataset = self[dataset_label]
        assert dataset is not None
        dataset.add_observation_to_index(observation.id)
        for element_label, observable_property_specification in labeled_observable_property_specifications.items():
            identifying = (
                str(observable_property_specification.specification_category)
                == peh.ObservablePropertySpecificationCategory.identifying.text
            )
            observable_property = observable_property_specification.observable_property
            assert isinstance(observable_property, peh.ObservableProperty)
            if identifying:
                if observable_property.id in dataset.schema._elements_by_observable_property:
                    self._register_observable_property(
                        observable_property_id=observable_property.id,
                        observation_id=observation.id,
                        dataset_label=dataset_label,
                        element_label=element_label,
                    )
                    continue
            _ = self.add_observable_property(
                observation_id=observation.id,
                observable_property_id=observable_property.id,
                data_type=ObservablePropertyValueType(getattr(observable_property, "value_type", "string")),
                dataset_label=dataset_label,
                element_label=element_label,
                is_primary_key=identifying,
            )

        if data is not None:
            dataset.add_data(data=data)

        return dataset

    #### RESTRUCTURE DATASETSERIES ####

    def subset_dataset(
        self,
        dataset_label: str,
        new_dataset_series_label: str,
        observation_groups: dict[str, list[peh.Observation]],
        observation_designs: list[peh.ObservationDesign],
        dataops_adapter: OutDataOpsInterface,
    ) -> DatasetSeries:
        """
        observation_groups: Contains the new `Dataset.label` as key, and the list[ObservationId] to be included in the new `Dataset`
        """

        dataset = self.get(dataset_label)
        assert dataset is not None
        return dataset.subset(
            new_dataset_series_label, observation_groups, observation_designs, dataops_adapter=dataops_adapter
        )

    def relabel_dataset(
        self, dataset_label: str, element_mapping: dict[str, str], dataops_adapter: OutDataOpsInterface
    ):
        dataset = self.get(dataset_label)
        assert dataset is not None
        return dataset.relabel(element_mapping, dataops_adapter=dataops_adapter)

    #### EXTRACT INFO FROM DATASETSERIES ####

    def get_type_annotations(self) -> dict[str, dict[str, ObservablePropertyValueType]]:
        ret: dict[str, dict[str, ObservablePropertyValueType]] = dict()
        for dataset in self.parts.values():
            label = dataset.label
            ret[label] = dataset.get_type_annotations()

        return ret

    def resolve_join(self, left_dataset_label: str, right_dataset_label: str) -> list[JoinSpec] | None:
        left = self.get(left_dataset_label)
        assert left is not None
        right = self.get(right_dataset_label)
        assert right is not None
        return left.resolve_join(right)

    def resolve_all_joins(self) -> dict[frozenset, list[JoinSpec] | None]:
        ret = {}
        for combo in itertools.combinations(self.parts, 2):
            key = frozenset(combo)
            ret[key] = self.resolve_join(*combo)
        return ret

    def matches_schema(self, raw_data_dict: dict[str, T_DataType], adapter: OutDataOpsInterface) -> bool:
        ret = True
        for dataset_label in self.parts:
            if dataset_label not in raw_data_dict:
                return False
            dataset = self[dataset_label]
            assert dataset is not None
            raw_data = raw_data_dict[dataset_label]
            raw_data_labels = set(adapter.get_element_labels(raw_data))
            ret = set(dataset.get_element_labels()) == raw_data_labels

        return ret

    def _get_validation_index(self) -> dict[str, dict[str, str]]:
        """
        # TEMPORARY SOLUTION !!!!
        ObservablePropertyId -> dataset_label, field_label
        We currently assume that each observable_property only occurs once in the DatasetSeries
        """
        field_ref_dict = defaultdict(dict)
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            for observable_property_id, element_labels in dataset.schema._elements_by_observable_property.items():
                assert (
                    len(element_labels) == 1
                ), "TEMPORARY RESTRICTION observable_properties should be unique within a dataset"
                element_label = next(iter(element_labels))
                field_ref_dict[observable_property_id][dataset_label] = element_label

        return field_ref_dict

    def get_contextual_field_reference_index(self) -> dict[str, tuple[str, str] | None]:
        field_ref_dict = {}
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            for observable_property_id, element_label in dataset.schema._elements_by_observable_property.items():
                if observable_property_id in field_ref_dict:
                    field_ref_dict[observable_property_id] = None
                else:
                    field_ref_dict[observable_property_id] = (dataset_label, element_label)

        return field_ref_dict

    def get_datasets_by_observation(self, observation_id: str) -> Generator[Dataset, None, None]:
        dataset_labels = self._obs_index.get(observation_id, None)
        if dataset_labels is None:
            return
        for dataset_label in dataset_labels:
            ret = self[dataset_label]
            assert ret is not None
            yield ret

    def context_lookup(self, observation_id: str, observable_property_id: str) -> tuple[str, str]:
        assert self._obs_index is not None, "Cannot perform lookup without observation index"
        assert self._context_index is not None, "Cannot perform lookup without context index"
        ret = self._context_index.get((observation_id, observable_property_id), None)
        if ret is None:
            raise ValueError(
                f"Could not find DatasetElement linked to observation with id {observation_id} and observable property with id {observable_property_id}"
            )

        return ret

    #### CORE FUNCTIONALITY ####

    @property
    def data_import_config(self) -> str | None:
        return self.metadata.get("data_import_config", None)

    @property
    def observations(self) -> dict[str, set[str]]:
        ret = {}
        for dataset_label in self:
            dataset = self[dataset_label]
            assert dataset is not None
            ret[dataset_label] = dataset.observation_ids

        return ret

    def __len__(self):
        return len(self.parts)

    def get(self, key, default=None) -> Dataset | None:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Dataset | None:
        return self.parts.get(key)

    def __setitem__(self, key: str, value: Dataset) -> None:
        self.parts[key] = value

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, " "got %d" % len(args))
            assert len(args) == 1
            other = dict(args[0])
            for key in other:
                self.parts[key] = other[key]
        for key in kwargs:
            self.parts[key] = kwargs[key]

    def __iter__(self):
        return iter(self.parts)


class MintingProtocol(Protocol):
    def mint_resource(self, resource_type: Type) -> str: ...


class ULIDFactory(MintingProtocol):
    def mint_resource(self, _: Type) -> str:
        return str(ULID())


class DatasetSeriesBuilder:
    def __init__(self, minting_protocol: MintingProtocol | None = ULIDFactory()):
        if minting_protocol is None:
            minting_protocol = ULIDFactory()
        self.minting_protocol = minting_protocol

    def mint_dataset_series(self, label: str, **kwargs):
        uri = self.minting_protocol.mint_resource(DatasetSeries)
        return DatasetSeries(
            label=label,
            identifier=uri,
            factory=self,
            **kwargs,
        )

    def mint_dataset(self, label: str, **kwargs):
        uri = self.minting_protocol.mint_resource(Dataset)
        return Dataset(
            label=label,
            identifier=uri,
            factory=self,
            **kwargs,
        )

    def mint_dataset_schema(self, **kwargs):
        uri = self.minting_protocol.mint_resource(DatasetSchema)
        return DatasetSchema(
            identifier=uri,
            factory=self,
            **kwargs,
        )

    def mint_dataset_schema_element(
        self, label: str, observable_property_id: str, data_type: ObservablePropertyValueType
    ):
        uri = self.minting_protocol.mint_resource(DatasetSchemaElement)
        return DatasetSchemaElement(
            label=label,
            identifier=uri,
            observable_property_id=observable_property_id,
            data_type=data_type,
        )

    def mint_foreign_key(self, element_label: str, reference: ElementReference):
        uri = self.minting_protocol.mint_resource(ForeignKey)
        return ForeignKey(
            identifier=uri,
            element_label=element_label,
            reference=reference,
        )

    def from_peh_datalayout(
        self,
        data_layout: peh.DataLayout,
        cache_view: CacheContainerView,
        apply_context: bool = True,
    ) -> DatasetSeries:
        dataset_series_label = data_layout.ui_label
        assert dataset_series_label is not None
        ret = self.mint_dataset_series(
            label=dataset_series_label,
            metadata={"described_by": data_layout.id},
        )
        sections = getattr(data_layout, "sections")
        if sections is None:
            raise ValueError("No sections found in DataLayout")
        for section in sections:
            dataset_label = getattr(section, "ui_label")
            dataset = ret.add_empty_dataset(dataset_label=dataset_label, metadata={"described_by": section.id})
            assert isinstance(section, peh.DataLayoutSection)
            section_elements = section.elements
            assert section_elements is not None
            for element in section_elements:
                assert isinstance(element, peh.DataLayoutElement)
                element_label = element.label
                assert element_label is not None
                observable_property_id = element.observable_property
                assert observable_property_id is not None
                identifying = getattr(element, "is_observable_entity_key", False)
                observation_id = str(ULID())
                observable_property = cache_view.get(observable_property_id, "ObservableProperty")
                assert isinstance(observable_property, peh.ObservableProperty)
                data_type = ObservablePropertyValueType(getattr(observable_property, "value_type", "string"))
                ret.add_observable_property(
                    observation_id=observation_id,
                    observable_property_id=observable_property_id,
                    data_type=data_type,
                    dataset_label=dataset_label,
                    element_label=element_label,
                    is_primary_key=identifying,
                )
                # ADD FOREIGN KEYS
                foreign_key = element.foreign_key_link
                if foreign_key is not None:
                    assert isinstance(foreign_key, peh.DataLayoutElementLink)
                    section_id = foreign_key.section
                    assert isinstance(section_id, str)
                    foreign_key_element_label = foreign_key.label
                    assert foreign_key_element_label is not None
                    section = cache_view.get(section_id, "DataLayoutSection")
                    assert section is not None, f"section with id {section_id} cannot be found"
                    foreign_key_dataset_label = section.ui_label
                    assert (
                        foreign_key_dataset_label is not None
                    ), f"Cannot create foreign_key_link, ui_label is None for section {section.id}"
                    dataset.schema.add_foreign_key_link(
                        element_label=element_label,
                        foreign_key_dataset_label=foreign_key_dataset_label,
                        foreign_key_element_label=foreign_key_element_label,
                    )

        if apply_context:
            ret.apply_context(cache=cache_view._container)

        return ret

    def from_peh_data_import_config(
        self,
        data_import_config: peh.DataImportConfig,
        cache_view: CacheContainerView,
        apply_context: bool = True,
    ) -> DatasetSeries:
        data_layout_id = data_import_config.layout
        assert data_layout_id is not None
        data_layout = cache_view.get(data_layout_id, "DataLayout")
        assert isinstance(data_layout, peh.DataLayout)
        layout_label = data_layout.ui_label
        if layout_label is None:
            layout_label = str(ULID())
        ret = self.mint_dataset_series(
            label=layout_label,
            metadata={"described_by": data_layout_id},
        )
        # add Observation links
        section_mapping = data_import_config.section_mapping
        assert isinstance(section_mapping, peh.DataImportSectionMapping)
        section_mapping_links = section_mapping.section_mapping_links
        assert isinstance(section_mapping_links, list)
        for link in section_mapping_links:
            assert isinstance(link, peh.DataImportSectionMappingLink)
            section_id = link.section
            assert isinstance(section_id, str)
            layout_section = cache_view.get(section_id, "DataLayoutSection")
            assert isinstance(layout_section, peh.DataLayoutSection)
            dataset_label = layout_section.ui_label
            assert dataset_label is not None
            dataset = ret.add_empty_dataset(
                dataset_label=dataset_label,
                metadata={"described_by": section_id},
            )
            observation_ids = link.observation_id_list
            assert observation_ids is not None
            assert isinstance(observation_ids, list)
            for observation_id in observation_ids:
                # observation_observable_properties
                observation = cache_view.get(observation_id, "Observation")
                assert isinstance(observation, peh.Observation)
                observation_design_id = observation.observation_design
                assert isinstance(observation_design_id, peh.ObservationDesignId)
                observation_design = cache_view.get(observation_design_id, "ObservationDesign")
                assert isinstance(observation_design, peh.ObservationDesign)
                obs_prop_spec_dict = {
                    str(s.observable_property): s for s in observation_design.observable_property_specifications
                }
                # data_layout_section_observable_properties
                elements = layout_section.elements
                assert elements is not None
                # labeled_observable_property_specifications = {element_label: ObservablePropertySpecification}
                labeled_observable_property_specifications = {}
                for element in elements:
                    assert isinstance(element, peh.DataLayoutElement)
                    if element.observable_property in obs_prop_spec_dict.keys():
                        element_label = element.label
                        assert element_label is not None
                        obs_prop_id = element.observable_property
                        assert isinstance(obs_prop_id, str)
                        obs_prop = cache_view.get(obs_prop_id, "ObservableProperty")
                        assert isinstance(obs_prop, peh.ObservableProperty)
                        obs_prop_spec = obs_prop_spec_dict[obs_prop_id]
                        assert isinstance(obs_prop_spec, peh.ObservablePropertySpecification)
                        obs_prop_spec.observable_property = obs_prop
                        labeled_observable_property_specifications[element_label] = obs_prop_spec
                        # process foreign keys
                        foreign_key = element.foreign_key_link
                        if foreign_key is not None:
                            assert isinstance(foreign_key, peh.DataLayoutElementLink)
                            section_id = foreign_key.section
                            assert isinstance(section_id, str)
                            foreign_key_element_label = foreign_key.label
                            assert foreign_key_element_label is not None
                            section = cache_view.get(section_id, "DataLayoutSection")
                            assert section is not None, f"section with id {section_id} cannot be found"
                            foreign_key_dataset_label = section.ui_label
                            assert (
                                foreign_key_dataset_label is not None
                            ), f"Cannot create foreign_key_link, ui_label is None for section {section.id}"
                            dataset.schema.add_foreign_key_link(
                                element_label=element_label,
                                foreign_key_dataset_label=foreign_key_dataset_label,
                                foreign_key_element_label=foreign_key_element_label,
                            )

                observation = cache_view.get(observation_id)
                assert isinstance(observation, peh.Observation)
                ret.add_observation(
                    dataset_label=dataset_label,
                    observation=observation,
                    labeled_observable_property_specifications=labeled_observable_property_specifications,
                )

        return ret
