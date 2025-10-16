"""
This module provides functionality for creating and manipulating an in-memory tree-like
representation of data.

Usage:
    Use this module to define in-memory data structures containing data adhering to the
    PEH-model.

"""

from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from collections import defaultdict
from peh_model.peh import NamedThing
from typing import Dict, Type, TYPE_CHECKING, Set, TypeVar, Generic

from pypeh.core.cache.utils import get_entity_type
from pypeh.core.models.proxy import TypedLazyProxy

if TYPE_CHECKING:
    from typing import Optional, Generator, Sequence
    from pypeh.core.models.typing import T_NamedThingLike

logger = logging.getLogger(__name__)

T_Container = TypeVar("T_Container")


class CacheContainer(ABC, Generic[T_Container]):
    """Abstract base class for cache backends"""

    def __init__(self):
        self._storage = T_Container

    @abstractmethod
    def add(self, entity: T_NamedThingLike) -> None:
        """Store an entity"""
        pass

    @abstractmethod
    def get(self, entity_id: str, entity_type: str | None = None) -> T_NamedThingLike:
        """Retrieve an entity"""
        pass

    @abstractmethod
    def get_all(self, entity_type: str | None = None) -> Generator[T_NamedThingLike, None, None]:
        """Retrieve all entities"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data"""
        pass

    @abstractmethod
    def exists(self, entity_id: str, entity_type: str | None = None) -> bool:
        """Clear all stored data"""
        pass

    @abstractmethod
    def pop(self, entity_id: str, entity_type: str) -> T_NamedThingLike:
        """Return entry and delete from cache"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class CacheContainerView(Generic[T_Container]):
    """Immutable view of any CacheContainer that only exposes read operations."""

    def __init__(self, container: CacheContainer[T_Container], container_subset: Sequence | None = None):
        self._container = container
        self.container_subset = container_subset

    def exists(self, entity_id: str, entity_type: str) -> bool:
        return self._container.exists(entity_id, entity_type)

    def get(self, entity_id: str, entity_type: str | None = None) -> Optional[T_NamedThingLike]:
        return self._container.get(entity_id, entity_type)

    def get_all(self, entity_type: str | None = None) -> Generator[T_NamedThingLike, None, None]:
        return self._container.get_all(entity_type)

    def iterate_subset(self) -> Generator[T_NamedThingLike, None, None]:
        if self.container_subset is not None:
            for entity_id in self.container_subset:
                ret = self.get(entity_id)
                if ret is not None:
                    yield ret
                else:
                    logger.debug(f"Entity with id {entity_id} not found in cache")
        else:
            logger.debug("No container_subset was provided")

    def __len__(self) -> int:
        return len(self._container)

    def __repr__(self):
        return f"ImmutableCacheContainerView({self._container!r})"


class MappingContainer(CacheContainer[Dict]):
    def __init__(self):
        self._storage: Dict[str, T_NamedThingLike] = dict()
        self._class_index: Dict[str, Set[str]] = defaultdict(set)

    def _add_object(self, entity: T_NamedThingLike, entity_id: str, entity_type: str) -> None:
        self._storage[entity_id] = entity
        self._class_index[entity_type].add(entity_id)

    def exists(self, entity_id: str, entity_type: str | None = None) -> bool:
        return entity_id in self._storage.keys()

    def _get(self, entity_id: str, entity_type: str | None = None) -> Optional[T_NamedThingLike]:
        if self.exists(entity_id, entity_type):
            return self._storage[entity_id]

    def add(self, entity: T_NamedThingLike) -> None:
        class_name = get_entity_type(entity)
        container_entity = self._get(entity.id, class_name)
        if container_entity is not None:
            if isinstance(container_entity, NamedThing):
                return
            if isinstance(entity, TypedLazyProxy):
                return
        return self._add_object(entity, entity.id, class_name)

    def get(self, entity_id: str, entity_type: str | None = None) -> Optional[T_NamedThingLike]:
        ret = self._get(entity_id, entity_type)
        if ret is None:
            message = f"Storage error: Object of class '{entity_type}' with id '{entity_id}' not found."
            logging.debug(message)
        return ret

    def clear(self) -> None:
        self._storage.clear()
        self._class_index.clear()

    def pop(self, entity_id: str, entity_type: str) -> Optional[T_NamedThingLike]:
        if entity_type in self._class_index:
            self._class_index[entity_type].remove(entity_id)
        return self._storage.pop(entity_id, None)

    def get_all(self, entity_type: str | None = None) -> Generator[T_NamedThingLike, None, None]:
        if entity_type is None:
            for entity_id in self._storage.keys():
                yield self._storage[entity_id]
        else:
            if entity_type in self._class_index:
                for entity_id in self._class_index[entity_type]:
                    yield self._storage[entity_id]

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self):
        return self._storage.__repr__()


class CacheContainerFactory:
    _default_container: Type[CacheContainer] = MappingContainer

    @classmethod
    def set_default_container(cls, container_class: Type[CacheContainer]):
        cls._default_container = container_class

    @classmethod
    def new(cls, container_type: str | None = None) -> CacheContainer:
        if container_type is not None:
            raise ValueError("Only a single Cachecontainer adapter has been implemented")
        return cls._default_container()
