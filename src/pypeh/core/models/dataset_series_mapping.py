from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Sequence

from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import (
    Dataset,
    DatasetSchema,
    DatasetSchemaElement,
    DatasetSeries,
)
from pypeh.core.models.typing import T_DataType


@dataclass(frozen=True)
class ObservablePropertyMapping:
    """
    Align one target ObservableProperty with one source property per input series.

    TODO: MIGRATE TO `peh_model`/LinkML.

    The order of `source_observable_property_ids` is positional by source
    collection. Direct concatenation expects exactly one source property for
    each source collection.
    """

    target_observable_property_id: str
    source_observable_property_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceObservationGroup:
    """
    Source Observations that contribute to a target Observation.

    TODO: MIGRATE TO `peh_model`/LinkML. Note that `peh_model`
    already has an ObservationGroup model.

    `source_observation_ids` may contain one or more source Observations. When
    resolving a scoped property mapping, the planner searches these Observations
    for the mapped source observable property. Multiple matches are accepted
    only when they resolve to the same concrete source field, which covers
    shared identifying fields registered for several Observations.
    """

    source_observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObservationAssembly:
    """
    Authoring model for one output Observation.

    TODO: MIGRATE TO `peh_model`/LinkML.

    `source_observation_groups` is positional by source collection. Each entry
    lists the source Observations that may contribute fields for this target
    Observation. `observable_property_mappings` is scoped to this target
    Observation, avoiding the old global cross-product between observations and
    properties. If no property alignments are provided, the planner infers an
    identity mapping for the shared observable properties resolvable in every
    source contribution.
    """

    target_observation_id: str
    source_observation_groups: tuple[SourceObservationGroup, ...]
    observable_property_mappings: tuple[ObservablePropertyMapping, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationAlignmentPlan:
    """
    Semantic alignment plan over Observations and ObservableProperties.

    TODO: MIGRATE TO `peh_model`/LinkML.

    This is the part of the model that can move toward `peh_model`: it does not
    name Datasets or DatasetSeries. Pypeh interprets source groups positionally
    against the concrete DatasetSeries supplied to concatenation.
    """

    observation_assemblies: tuple[ObservationAssembly, ...]


@dataclass(frozen=True)
class DatasetSeriesAlignment:
    """
    Pypeh wrapper for applying an ObservationAlignmentPlan to DatasetSeries.

    Session-level APIs accept `ObservationAlignmentPlan` directly and construct
    this wrapper internally to attach DatasetSeries-specific options such as the
    output label.

    Example for two source series:

        DatasetSeriesAlignment(
            alignment_plan=ObservationAlignmentPlan(
                observation_assemblies=(
                    ObservationAssembly(
                        target_observation_id="peh:obs_lab",
                        source_observation_groups=(
                            SourceObservationGroup(("study_a:sample",)),
                            SourceObservationGroup(("study_b:subject",)),
                        ),
                        observable_property_mappings=(
                            ObservablePropertyMapping(
                                target_observable_property_id="peh:chol",
                                source_observable_property_ids=(
                                    "study_a:chol",
                                    "study_b:total_cholesterol",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    Restrictions for direct concatenation:
    - each source tuple must have one entry per input DatasetSeries;
    - each scoped property must resolve to exactly one source field per series;
    - fields aligned into one output element must have compatible value types;
    - one output dataset cannot be assembled from multiple source datasets
      within the same source series.
    """

    alignment_plan: ObservationAlignmentPlan
    output_label: str | None = None

    def with_output_label(
        self, output_label: str | None
    ) -> DatasetSeriesAlignment:
        if output_label is None:
            return self
        return DatasetSeriesAlignment(
            alignment_plan=self.alignment_plan,
            output_label=output_label,
        )


@dataclass(frozen=True)
class DatasetSeriesRef:
    """Reference to one source DatasetSeries in a generated concatenation plan."""

    series_index: int
    series_label: str


@dataclass(frozen=True)
class DatasetRef:
    """Reference to one Dataset inside one source DatasetSeries."""

    series_index: int
    series_label: str
    dataset_label: str


@dataclass(frozen=True)
class ContextualElementRef:
    """
    Resolved source field for one aligned observation/property pair.

    These are normally created by `DatasetSeriesConcatenationPlan.from_alignment`
    after resolving a user-authored `DatasetSeriesAlignment` through the source
    context indices.
    """

    target_observation_id: str
    source_observation_id: str
    source_observable_property_id: str
    dataset: DatasetRef
    element_label: str


@dataclass(frozen=True)
class ElementConcatenationMapping:
    """
    Generated mapping from resolved source fields to one output element.

    This is a projection/relabeling mapping, not a transformation mapping. The
    listed sources are expected to represent the same output observable property
    and to have the same value type.
    """

    output_element_label: str
    output_observable_property_id: str
    sources: tuple[ContextualElementRef, ...]
    value_type: ObservablePropertyValueType

    def with_sources(
        self, sources: Sequence[ContextualElementRef]
    ) -> ElementConcatenationMapping:
        return ElementConcatenationMapping(
            output_element_label=self.output_element_label,
            output_observable_property_id=self.output_observable_property_id,
            sources=tuple(sources),
            value_type=self.value_type,
        )


@dataclass(frozen=True)
class DatasetConcatenationMapping:
    """
    Generated mapping for one output Dataset in a concatenation plan.

    `sources` contains one source Dataset per input DatasetSeries. Direct
    concatenation currently cannot build the same output Dataset from multiple
    source Datasets within a single source series.
    """

    output_dataset_label: str
    sources: tuple[DatasetRef, ...]
    elements: tuple[ElementConcatenationMapping, ...]
    output_observation_ids: tuple[str, ...] = ()

    def build_output_schema(
        self,
        source_dataset: Dataset[T_DataType],
    ) -> DatasetSchema:
        assert source_dataset.schema is not None
        output_labels = {
            element_mapping.output_element_label
            for element_mapping in self.elements
        }
        elements = {
            element_mapping.output_element_label: DatasetSchemaElement(
                label=element_mapping.output_element_label,
                observable_property_id=(
                    element_mapping.output_observable_property_id
                ),
                data_type=element_mapping.value_type,
            )
            for element_mapping in self.elements
        }
        primary_keys = {
            element_mapping.output_element_label
            for element_mapping in self.elements
            if element_mapping.sources[0].element_label
            in source_dataset.schema.primary_keys
        }
        foreign_keys = {
            label: foreign_key
            for label, foreign_key in source_dataset.schema.foreign_keys.items()
            if label in output_labels
        }
        return DatasetSchema(
            elements=elements,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
        )


@dataclass(frozen=True)
class DatasetSeriesConcatenationPlan(Generic[T_DataType]):
    output_label: str
    sources: tuple[DatasetSeriesRef, ...]
    datasets: tuple[DatasetConcatenationMapping, ...]

    @staticmethod
    def _schema_elements_by_observable_property(
        dataset: Dataset[T_DataType],
    ) -> dict[str, DatasetSchemaElement]:
        if dataset.schema is None:
            raise ValueError(f"Dataset {dataset.label!r} has no schema.")
        ret: dict[str, DatasetSchemaElement] = {}
        for element in dataset.schema.elements.values():
            observable_property_id = element.observable_property_id
            element_labels = dataset.schema._elements_by_observable_property[
                observable_property_id
            ]
            if len(element_labels) != 1:
                raise ValueError(
                    f"Dataset {dataset.label!r} contains observable property "
                    f"{observable_property_id!r} more than once; strict "
                    "concatenation currently requires one element per "
                    "observable property per dataset."
                )
            ret[observable_property_id] = element
        return ret

    @staticmethod
    def _validate_property_alignment(
        alignment: ObservablePropertyMapping,
        *,
        source_count: int,
    ) -> None:
        if len(alignment.source_observable_property_ids) != source_count:
            raise ValueError(
                "ObservablePropertyMapping "
                f"{alignment.target_observable_property_id!r} must contain "
                f"{source_count} source observable property ids; found "
                f"{len(alignment.source_observable_property_ids)}."
            )

    @classmethod
    def _observation_assemblies(
        cls,
        alignment: DatasetSeriesAlignment,
        source_count: int,
    ) -> tuple[ObservationAssembly, ...]:
        return alignment.alignment_plan.observation_assemblies

    @classmethod
    def _validate_target_alignments(
        cls,
        target_alignments: Sequence[ObservationAssembly],
        source_count: int,
    ) -> None:
        if len(target_alignments) == 0:
            raise ValueError(
                "DatasetSeriesAlignment must contain at least one target "
                "observation alignment."
            )
        for target_alignment in target_alignments:
            if len(target_alignment.source_observation_groups) != source_count:
                raise ValueError(
                    "ObservationAssembly "
                    f"{target_alignment.target_observation_id!r} must "
                    f"contain {source_count} source contributions; found "
                    f"{len(target_alignment.source_observation_groups)}."
                )
            for series_index, contribution in enumerate(
                target_alignment.source_observation_groups
            ):
                if len(contribution.source_observation_ids) == 0:
                    raise ValueError(
                        "ObservationAssembly "
                        f"{target_alignment.target_observation_id!r} contains "
                        "an empty source observation contribution for series "
                        f"index {series_index}."
                    )
            for (
                property_alignment
            ) in target_alignment.observable_property_mappings:
                cls._validate_property_alignment(
                    property_alignment,
                    source_count=source_count,
                )

    @staticmethod
    def _resolve_contextual_source(
        series: DatasetSeries[T_DataType],
        *,
        target_observation_id: str,
        source_observation_ids: Sequence[str],
        source_property_id: str,
    ) -> tuple[str, str, str]:
        resolved_refs: dict[tuple[str, str], str] = {}
        for source_observation_id in source_observation_ids:
            try:
                contextual_ref = series.context_lookup(
                    source_observation_id,
                    source_property_id,
                )
            except ValueError:
                continue
            resolved_refs.setdefault(contextual_ref, source_observation_id)

        if len(resolved_refs) == 0:
            raise ValueError(
                "Could not resolve observable property "
                f"{source_property_id!r} for target observation "
                f"{target_observation_id!r} in series {series.label!r}; "
                f"tried source observations {tuple(source_observation_ids)!r}."
            )
        if len(resolved_refs) > 1:
            raise ValueError(
                "Observable property "
                f"{source_property_id!r} for target observation "
                f"{target_observation_id!r} resolves to multiple source "
                f"fields in series {series.label!r}: "
                f"{sorted(resolved_refs)!r}."
            )
        contextual_ref, source_observation_id = next(
            iter(resolved_refs.items())
        )
        source_dataset_label, source_element_label = contextual_ref
        return (
            source_observation_id,
            source_dataset_label,
            source_element_label,
        )

    @staticmethod
    def _resolvable_property_ids(
        series: DatasetSeries[T_DataType],
        source_observation_ids: Sequence[str],
    ) -> set[str]:
        ret: set[str] = set()
        for (
            observation_id,
            observable_property_id,
        ) in series._context_index:
            if observation_id in source_observation_ids:
                ret.add(observable_property_id)
        return ret

    @staticmethod
    def _ordered_resolvable_property_ids(
        series: DatasetSeries[T_DataType],
        source_observation_ids: Sequence[str],
    ) -> tuple[str, ...]:
        ret: list[str] = []
        seen: set[str] = set()
        for (
            observation_id,
            observable_property_id,
        ) in series._context_index:
            if (
                observation_id in source_observation_ids
                and observable_property_id not in seen
            ):
                seen.add(observable_property_id)
                ret.append(observable_property_id)
        return tuple(ret)

    @classmethod
    def _infer_property_alignments(
        cls,
        dataset_series: Sequence[DatasetSeries[T_DataType]],
        target_alignment: ObservationAssembly,
    ) -> tuple[ObservablePropertyMapping, ...]:
        shared_property_ids: set[str] | None = None
        for series_index, series in enumerate(dataset_series):
            contribution = target_alignment.source_observation_groups[
                series_index
            ]
            property_ids = cls._resolvable_property_ids(
                series,
                contribution.source_observation_ids,
            )
            if shared_property_ids is None:
                shared_property_ids = property_ids
            else:
                shared_property_ids &= property_ids

        if not shared_property_ids:
            raise ValueError(
                "Could not infer observable property alignments for target "
                f"observation {target_alignment.target_observation_id!r}; "
                "source contributions have no shared observable properties."
            )

        canonical_property_ids = cls._ordered_resolvable_property_ids(
            dataset_series[0],
            target_alignment.source_observation_groups[
                0
            ].source_observation_ids,
        )
        return tuple(
            ObservablePropertyMapping(
                target_observable_property_id=observable_property_id,
                source_observable_property_ids=tuple(
                    observable_property_id for _ in dataset_series
                ),
            )
            for observable_property_id in canonical_property_ids
            if observable_property_id in shared_property_ids
        )

    @classmethod
    def from_alignment(
        cls,
        dataset_series: Sequence[DatasetSeries[T_DataType]],
        alignment: DatasetSeriesAlignment,
        *,
        output_label: str | None = None,
    ) -> DatasetSeriesConcatenationPlan[T_DataType]:
        if len(dataset_series) == 0:
            raise ValueError(
                "Cannot concatenate an empty sequence of DatasetSeries."
            )
        source_count = len(dataset_series)
        target_alignments = cls._observation_assemblies(
            alignment,
            source_count,
        )
        cls._validate_target_alignments(target_alignments, source_count)

        first = dataset_series[0]
        output_label = (
            output_label
            or alignment.output_label
            or f"{first.label}_concatenated"
        )
        source_series_refs = tuple(
            DatasetSeriesRef(
                series_index=series_index,
                series_label=series.label,
            )
            for series_index, series in enumerate(dataset_series)
        )

        dataset_refs_by_output: dict[str, dict[int, DatasetRef]] = {}
        elements_by_output: dict[
            str, dict[str, ElementConcatenationMapping]
        ] = {}
        observation_ids_by_output: dict[str, set[str]] = {}

        for target_alignment in target_alignments:
            property_alignments = (
                target_alignment.observable_property_mappings
                or cls._infer_property_alignments(
                    dataset_series,
                    target_alignment,
                )
            )
            for property_alignment in property_alignments:
                canonical_property_id = (
                    property_alignment.source_observable_property_ids[0]
                )
                (
                    _,
                    canonical_dataset_label,
                    canonical_element_label,
                ) = cls._resolve_contextual_source(
                    first,
                    target_observation_id=(
                        target_alignment.target_observation_id
                    ),
                    source_observation_ids=(
                        target_alignment.source_observation_groups[
                            0
                        ].source_observation_ids
                    ),
                    source_property_id=canonical_property_id,
                )
                canonical_dataset = first.parts[canonical_dataset_label]
                canonical_schema_element = (
                    canonical_dataset.get_schema_element_by_label(
                        canonical_element_label
                    )
                )
                if canonical_schema_element is None:
                    raise ValueError(
                        f"Could not find canonical schema element "
                        f"{canonical_element_label!r} in dataset "
                        f"{canonical_dataset_label!r}."
                    )

                source_refs: list[ContextualElementRef] = []
                for series_index, series in enumerate(dataset_series):
                    source_property_id = (
                        property_alignment.source_observable_property_ids[
                            series_index
                        ]
                    )
                    (
                        source_observation_id,
                        source_dataset_label,
                        source_element_label,
                    ) = cls._resolve_contextual_source(
                        series,
                        target_observation_id=(
                            target_alignment.target_observation_id
                        ),
                        source_observation_ids=(
                            target_alignment.source_observation_groups[
                                series_index
                            ].source_observation_ids
                        ),
                        source_property_id=source_property_id,
                    )
                    source_dataset = series.parts[source_dataset_label]
                    source_schema_element = (
                        source_dataset.get_schema_element_by_label(
                            source_element_label
                        )
                    )
                    if source_schema_element is None:
                        raise ValueError(
                            f"Could not find schema element "
                            f"{source_element_label!r} in dataset "
                            f"{source_dataset_label!r} for series "
                            f"{series.label!r}."
                        )
                    if (
                        source_schema_element.data_type
                        != canonical_schema_element.data_type
                    ):
                        raise ValueError(
                            "Aligned observable properties require "
                            "compatible value types. Expected "
                            f"{canonical_schema_element.data_type!r}, found "
                            f"{source_schema_element.data_type!r} for "
                            f"observable property {source_property_id!r} in "
                            f"series {series.label!r}."
                        )
                    dataset_ref = DatasetRef(
                        series_index=series_index,
                        series_label=series.label,
                        dataset_label=source_dataset_label,
                    )
                    dataset_refs_for_output = (
                        dataset_refs_by_output.setdefault(
                            canonical_dataset_label, {}
                        )
                    )
                    existing_dataset_ref = dataset_refs_for_output.get(
                        series_index
                    )
                    if (
                        existing_dataset_ref is not None
                        and existing_dataset_ref != dataset_ref
                    ):
                        raise ValueError(
                            "Alignment resolves output dataset "
                            f"{canonical_dataset_label!r} to multiple source "
                            f"datasets for series {series.label!r}: "
                            f"{existing_dataset_ref.dataset_label!r} and "
                            f"{dataset_ref.dataset_label!r}. This is not "
                            "supported by minimal concatenation."
                        )
                    dataset_refs_for_output[series_index] = dataset_ref
                    source_refs.append(
                        ContextualElementRef(
                            target_observation_id=(
                                target_alignment.target_observation_id
                            ),
                            source_observation_id=source_observation_id,
                            source_observable_property_id=source_property_id,
                            dataset=dataset_ref,
                            element_label=source_element_label,
                        )
                    )

                output_property_id = (
                    property_alignment.target_observable_property_id
                )
                output_elements = elements_by_output.setdefault(
                    canonical_dataset_label, {}
                )
                observation_ids_by_output.setdefault(
                    canonical_dataset_label, set()
                ).add(target_alignment.target_observation_id)
                existing = output_elements.get(output_property_id)
                element_mapping = ElementConcatenationMapping(
                    output_element_label=canonical_element_label,
                    output_observable_property_id=output_property_id,
                    sources=tuple(source_refs),
                    value_type=canonical_schema_element.data_type,
                )
                if existing is not None:
                    if (
                        existing.output_element_label
                        != element_mapping.output_element_label
                        or existing.value_type != element_mapping.value_type
                    ):
                        raise ValueError(
                            "Alignment resolves observable property "
                            f"{output_property_id!r} to multiple incompatible "
                            "output fields in dataset "
                            f"{canonical_dataset_label!r}."
                        )
                    element_mapping = existing.with_sources(
                        (*existing.sources, *element_mapping.sources)
                    )
                output_elements[output_property_id] = element_mapping

        dataset_mappings: list[DatasetConcatenationMapping] = []
        for output_dataset_label in sorted(elements_by_output):
            source_refs_by_series = dataset_refs_by_output[
                output_dataset_label
            ]
            missing_series = [
                series.label
                for series_index, series in enumerate(dataset_series)
                if series_index not in source_refs_by_series
            ]
            if len(missing_series) > 0:
                raise ValueError(
                    f"Dataset alignment for output dataset "
                    f"{output_dataset_label!r} does not include every source "
                    f"series. Missing {missing_series!r}."
                )
            dataset_mappings.append(
                DatasetConcatenationMapping(
                    output_dataset_label=output_dataset_label,
                    sources=tuple(
                        source_refs_by_series[series_index]
                        for series_index in range(len(dataset_series))
                    ),
                    elements=tuple(
                        elements_by_output[output_dataset_label].values()
                    ),
                    output_observation_ids=tuple(
                        sorted(observation_ids_by_output[output_dataset_label])
                    ),
                )
            )

        return cls(
            output_label=output_label,
            sources=source_series_refs,
            datasets=tuple(dataset_mappings),
        )

    @classmethod
    def from_strict_dataset_series(
        cls,
        dataset_series: Sequence[DatasetSeries[T_DataType]],
        *,
        output_label: str | None = None,
    ) -> DatasetSeriesConcatenationPlan[T_DataType]:
        if len(dataset_series) == 0:
            raise ValueError(
                "Cannot concatenate an empty sequence of DatasetSeries."
            )

        first = dataset_series[0]
        expected_dataset_labels = set(first.parts.keys())
        for series in dataset_series[1:]:
            dataset_labels = set(series.parts.keys())
            if dataset_labels != expected_dataset_labels:
                raise ValueError(
                    "Strict DatasetSeries concatenation requires identical "
                    "dataset labels. Expected "
                    f"{sorted(expected_dataset_labels)!r}, found "
                    f"{sorted(dataset_labels)!r} for series "
                    f"{series.label!r}."
                )

        dataset_mappings: list[DatasetConcatenationMapping] = []
        for dataset_label in sorted(expected_dataset_labels):
            canonical_dataset = first.parts[dataset_label]
            canonical_elements = cls._schema_elements_by_observable_property(
                canonical_dataset
            )
            expected_property_id_order = list(canonical_elements.keys())
            expected_property_ids = set(expected_property_id_order)
            dataset_refs = tuple(
                DatasetRef(
                    series_index=series_index,
                    series_label=series.label,
                    dataset_label=dataset_label,
                )
                for series_index, series in enumerate(dataset_series)
            )
            element_mappings: list[ElementConcatenationMapping] = []

            for observable_property_id in expected_property_id_order:
                canonical_element = canonical_elements[observable_property_id]
                source_refs: list[ContextualElementRef] = []
                for dataset_ref in dataset_refs:
                    source_dataset = dataset_series[
                        dataset_ref.series_index
                    ].parts[dataset_label]
                    source_elements = (
                        cls._schema_elements_by_observable_property(
                            source_dataset
                        )
                    )
                    source_property_ids = set(source_elements.keys())
                    if source_property_ids != expected_property_ids:
                        raise ValueError(
                            "Strict DatasetSeries concatenation requires "
                            "identical observable property ids for dataset "
                            f"{dataset_label!r}. Expected "
                            f"{sorted(expected_property_ids)!r}, found "
                            f"{sorted(source_property_ids)!r} for series "
                            f"{dataset_ref.series_label!r}."
                        )
                    source_element = source_elements[observable_property_id]
                    if source_element.data_type != canonical_element.data_type:
                        raise ValueError(
                            "Strict DatasetSeries concatenation requires "
                            "compatible value types for observable property "
                            f"{observable_property_id!r} in dataset "
                            f"{dataset_label!r}. Expected "
                            f"{canonical_element.data_type!r}, found "
                            f"{source_element.data_type!r} for series "
                            f"{dataset_ref.series_label!r}."
                        )
                    for observation_id in sorted(
                        source_dataset.observation_ids
                    ):
                        source_refs.append(
                            ContextualElementRef(
                                target_observation_id=observation_id,
                                source_observation_id=observation_id,
                                source_observable_property_id=(
                                    observable_property_id
                                ),
                                dataset=dataset_ref,
                                element_label=source_element.label,
                            )
                        )

                element_mappings.append(
                    ElementConcatenationMapping(
                        output_element_label=canonical_element.label,
                        output_observable_property_id=observable_property_id,
                        sources=tuple(source_refs),
                        value_type=canonical_element.data_type,
                    )
                )

            dataset_mappings.append(
                DatasetConcatenationMapping(
                    output_dataset_label=dataset_label,
                    sources=dataset_refs,
                    elements=tuple(element_mappings),
                )
            )

        return cls(
            output_label=output_label or f"{first.label}_concatenated",
            sources=tuple(
                DatasetSeriesRef(
                    series_index=series_index,
                    series_label=series.label,
                )
                for series_index, series in enumerate(dataset_series)
            ),
            datasets=tuple(dataset_mappings),
        )
