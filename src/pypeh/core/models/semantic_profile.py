from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from rdflib import URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, DCTERMS, XSD, PROV
from typing import Iterable, Type
from ulid import ULID

from pypeh.core.models.internal_data_layout import Resource, Dataset, DatasetSeries, DatasetSchema
from pypeh.core.models.constants import ObservablePropertyValueType

PEH = Namespace("https://w3id.org/peh/terms#")
CSVW = Namespace("http://www.w3.org/ns/csvw#")
DCAT = Namespace("http://www.w3.org/ns/dcat#")  # ensure DCAT3 is used

XSD_TYPE_MAP: dict[ObservablePropertyValueType, URIRef] = {
    ObservablePropertyValueType.INTEGER: XSD.integer,
    ObservablePropertyValueType.FLOAT: XSD.double,
    ObservablePropertyValueType.STRING: XSD.string,
    ObservablePropertyValueType.BOOLEAN: XSD.boolean,
    ObservablePropertyValueType.DATETIME: XSD.dateTime,
    ObservablePropertyValueType.DATE: XSD.date,
}


@dataclass(kw_only=True)
class SemanticDatasetSchema(DatasetSchema):
    @abstractmethod
    def to_rdf(self, dataset: SemanticDataset):
        raise NotImplementedError

    @classmethod
    def from_dataset_schema(cls, dataset_schema: DatasetSchema) -> SemanticDatasetSchema:
        return cls(
            elements=dataset_schema.elements,
            primary_keys=dataset_schema.primary_keys,
            foreign_keys=dataset_schema.foreign_keys,
        )


@dataclass
class CSVWDatasetSchema(SemanticDatasetSchema):
    """
    Extends DatasetSchema with CSVW serialization.

    Emits a csvw:TableSchema with typed columns, primary keys, and
    foreign keys. Each column links its PEH ObservableProperty via
    both csvw:propertyUrl (CSVW consumers) and
    peh:hasObservableProperty (PEH consumers). Datatypes are mapped
    to proper xsd: URIs.
    """

    def to_rdf(self, dataset: SemanticDataset) -> Iterable[tuple]:
        s = URIRef(dataset.identifier)

        # TableSchema node anchored to the Dataset URI
        schema_node = BNode()
        yield (s, CSVW.tableSchema, schema_node)
        yield (schema_node, RDF.type, CSVW.Schema)

        # Columns
        for label, element in self.elements.items():
            col_node = BNode()
            yield (schema_node, CSVW.column, col_node)
            yield (col_node, RDF.type, CSVW.Column)
            yield (col_node, CSVW.name, Literal(label))

            xsd_type = XSD_TYPE_MAP.get(element.data_type)
            if xsd_type is None:
                raise ValueError(
                    f"No XSD type mapping for {element.data_type!r} " f"on column '{label}'. Add it to XSD_TYPE_MAP."
                )
            yield (col_node, CSVW.datatype, xsd_type)

            # dual linking to ObservableProperty
            if element.observable_property_id:
                op = URIRef(element.observable_property_id)
                yield (col_node, CSVW.propertyUrl, op)
                yield (col_node, PEH.hasObservableProperty, op)

        # Primary keys — one triple per key (more SPARQL-friendly
        # than the spec's space-separated single Literal)
        for key in self.primary_keys:
            yield (schema_node, CSVW.primaryKey, Literal(key))

        # Foreign keys
        for fk_label, fk in self.foreign_keys.items():
            fk_node = BNode()
            ref_node = BNode()
            yield (schema_node, CSVW.foreignKey, fk_node)
            yield (fk_node, CSVW.columnReference, Literal(fk.element_label))
            yield (fk_node, CSVW.reference, ref_node)
            yield (ref_node, CSVW.resource, Literal(fk.reference.dataset_label))
            yield (ref_node, CSVW.columnReference, Literal(fk.reference.element_label))


@dataclass(kw_only=True)
class SemanticResource(Resource):
    """
    Base class for Dataset and DatasetSeries.
    Serializes to dcat:Resource triples.
    """

    label: str
    identifier: str = field(default_factory=lambda: str(ULID()))

    was_attributed_to: str | None = field(default=None, metadata={"id": "prov:wasAttributedTo"})
    creator: str | None = field(default=None, metadata={"id": "dcterms:creator"})
    created: str | None = field(default=None, metadata={"id": "dcterms:created"})
    modified: str | None = field(default=None, metadata={"id": "dcterms:modified"})
    issued: str | None = field(default=None, metadata={"id": "dcterms:issued"})
    description: str | None = field(default=None, metadata={"id": "dcterms:description"})

    @classmethod
    def from_resource(cls, resource: Resource, **kwargs):
        return cls(label=resource.label, identifier=resource.identifier, **kwargs)

    def to_rdf(self) -> Iterable[tuple]:
        s = URIRef(self.identifier)

        yield (s, DCTERMS.title, Literal(self.label))
        yield (s, DCTERMS.identifier, Literal(self.identifier))

        if self.description:
            yield (s, DCTERMS.description, Literal(self.description))
        if self.issued:
            yield (s, DCTERMS.issued, Literal(self.issued, datatype=XSD.dateTime))
        if self.modified:
            yield (s, DCTERMS.modified, Literal(self.modified, datatype=XSD.dateTime))
        if self.created:
            yield (s, DCTERMS.created, Literal(self.created, datatype=XSD.dateTime))
        if self.creator:
            yield (s, DCTERMS.creator, URIRef(self.creator))
        if self.was_attributed_to:
            yield (s, PROV.wasAttributedTo, URIRef(self.was_attributed_to))


@dataclass(kw_only=True)
class SemanticDataset(SemanticResource):
    """
    Serializes to dcat:Dataset + prov:Entity triples.
    Observations link out to the peh ontology.
    """

    schema: SemanticDatasetSchema = field(default_factory=SemanticDatasetSchema)
    data: str | None = field(default=None)
    part_of: SemanticDatasetSeries | DatasetSeries | None = field(default=None)
    observation_ids: set[str] = field(default_factory=set)  # URIs of peh:Observation instances

    # PROV — explicit typed fields
    was_generated_by: str | None = field(default=None)  # prov:wasGeneratedBy   (activity URI)
    was_derived_from: str | None = field(default=None)  # prov:wasDerivedFrom   (entity URI)

    @classmethod
    def from_dataset(cls, dataset: Dataset, schema_profile: Type[SemanticDatasetSchema], **kwargs):
        return cls(
            label=dataset.label,
            identifier=dataset.identifier,
            schema=schema_profile.from_dataset_schema(dataset.schema),
            part_of=dataset.part_of,
            observation_ids=dataset.observation_ids,
            **kwargs,
        )

    def to_rdf(self) -> Iterable[tuple]:
        s = URIRef(self.identifier)

        # types
        yield (s, RDF.type, DCAT.Dataset)
        yield (s, RDF.type, PROV.Entity)

        # inherited Resource triples (DCAT + PROV base)
        yield from super().to_rdf()

        # dataset-level PROV
        if self.was_generated_by:
            yield (s, PROV.wasGeneratedBy, URIRef(self.was_generated_by))
        if self.was_derived_from:
            yield (s, PROV.wasDerivedFrom, URIRef(self.was_derived_from))

        # series membership
        if self.part_of is not None:
            yield (s, DCTERMS.isPartOf, URIRef(self.part_of.identifier))
            yield (s, DCAT.inSeries, URIRef(self.part_of.identifier))

        # peh observation_ids — each URI points to a peh:Observation owl:namedIndividual
        for obs_uri in self.observation_ids:
            yield (s, PEH.hasObservation, URIRef(obs_uri))


@dataclass(kw_only=True)
class SemanticDatasetSeries(SemanticResource):
    """
    Serializes to dcat:DatasetSeries triples.
    Parts are referenced by URI only (no nested serialization).
    """

    parts: list[str] = field(default_factory=list)

    @classmethod
    def from_dataset_series(cls, dataset_series: DatasetSeries, **kwargs):
        return cls(
            label=dataset_series.label,
            identifier=dataset_series.identifier,
            parts=list(dataset.identifier for dataset in dataset_series.parts.values()),
            **kwargs,
        )

    def to_rdf(self) -> Iterable[tuple]:
        s = URIRef(self.identifier)

        # types
        yield (s, RDF.type, DCAT.DatasetSeries)
        yield (s, RDF.type, PROV.Entity)

        # inherited Resource triples
        yield from super().to_rdf()

        # member references only — each Dataset serializes itself independently
        for dataset_id in self.parts:
            yield (s, DCAT.seriesMember, URIRef(dataset_id))
