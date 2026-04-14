"""
graph/base.py
그래프 백엔드 추상 인터페이스
NetworkX / Neo4j 모두 이 인터페이스를 구현
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class GraphBackend(ABC):

    @abstractmethod
    def merge_entity(self, label: str, name: str) -> None:
        """엔티티 추가 (중복 무시)"""

    @abstractmethod
    def merge_relation(self, source: str, relation: str, target: str) -> None:
        """관계 추가 (중복 무시)"""

    @abstractmethod
    def search_nodes(self, query: str, limit: int = 10) -> List[Dict]:
        """이름에 query가 포함된 노드 검색"""

    @abstractmethod
    def search_related(self, name: str, depth: int = 2) -> List[Dict]:
        """연결 관계 탐색"""

    @abstractmethod
    def get_stats(self) -> Dict:
        """노드/관계 수 통계"""

    @abstractmethod
    def get_labels(self) -> List[str]:
        """모든 노드 레이블"""

    @abstractmethod
    def clear(self) -> None:
        """전체 초기화"""

    @abstractmethod
    def test_connection(self) -> bool:
        """연결/가용 여부"""
