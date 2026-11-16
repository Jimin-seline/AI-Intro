"""정보 없는 탐색 (담당: 서지민). 현재 항의만 반영한 결정적 모델에서 계획한다.

UCS: g(누적 비용)가 가장 작은 상태부터 펼친다. 휴리스틱이 없는 A*와 같고,
     다른 알고리즘의 최적격차 기준 해가 된다 (L0·L1).
"""
from collections import deque
from heapq import heappop, heappush
from itertools import count
from math import inf

from env import Agent

OBJECTIVES = ("c1", "c2")   # c1 = ① 총비용, c2 = ② 완료 시각 합


def build_plan(parents, state):
    """목표 상태에서 부모를 따라 올라가 행동 목록을 만든다."""
    actions = deque()
    while parents[state] is not None:
        state, action = parents[state]
        actions.appendleft(action)
    return actions


class UCSAgent(Agent):
    name = "ucs"

    def __init__(self, replan="never", objective="c1"):
        super().__init__(replan)
        if objective not in OBJECTIVES:
            raise ValueError(f"objective는 {OBJECTIVES} 중 하나여야 한다")
        self.objective = objective

    def reset(self, env, rng):
        super().reset(env, rng)
        self._plan = deque()

    def choose_action(self, obs):
        if self.needs_replan(obs):
            self._plan = self._search(obs)
        if not self._plan:
            raise RuntimeError("UCS 계획이 비어 있습니다. 목표 상태 또는 도달 가능성을 확인하세요.")
        return self._plan.popleft()

    def _search(self, start):
        serial = count()                        # 같은 g면 먼저 넣은 상태부터 (결과 재현용)
        frontier = [(0, next(serial), start)]
        best_g = {start: 0}
        parents = {start: None}
        while frontier:
            self.report("mem_states", len(frontier) + len(best_g))
            self.env._check_deadline()
            g, _, state = heappop(frontier)
            if g != best_g[state]:              # 더 싼 경로가 이미 나온 오래된 항목
                continue
            if self.env.is_goal(state):         # 꺼낼 때 목표 검사해야 최적이 보장된다
                plan = build_plan(parents, state)
                self.report("plan_len", len(plan))
                return plan
            for action, outcomes in self.env.successors(state, new_complaints=False):
                _, nxt, cost = outcomes[0]
                candidate = g + getattr(cost, self.objective)
                if candidate >= best_g.get(nxt, inf):
                    continue
                best_g[nxt] = candidate
                parents[nxt] = (state, action)
                heappush(frontier, (candidate, next(serial), nxt))
        raise RuntimeError("UCS 탐색에서 배달 완료 경로를 찾지 못했습니다.")
