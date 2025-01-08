# COSMOS (Convex Optimization for Slicing with Multiple Objective Scheduling)

import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

@dataclass
class SliceProblem:
    """슬라이스별 최적화 문제를 위한 데이터 클래스"""
    slice_id: int
    spectral_efficiency: float
    queue_length: float
    slice_type: str

class COSMOS:
    """
    COSMOS (Coordinated Optimization for Slicing with Multiple Objective Scheduling)
    Dual Decomposition과 Primal-Dual 접근법을 활용한 Inter-slice 자원 할당 최적화
    """
    def __init__(
        self, 
        total_rbs: int,
        num_slices: int,
        weights: Dict[str, float],
        learning_rate: float = 0.01,
        max_iter: int = 100,
        eps: float = 1e-6,
        alpha: float = 1.0,
        min_latency: float = 0.1,
        convergence_threshold: float = 1e-4
    ):
        """
        COSMOS 옵티마이저 초기화
        
        Args:
            total_rbs: 총 가용 리소스 블록 수
            num_slices: 네트워크 슬라이스 수
            weights: 슬라이스별 목적함수 가중치
            learning_rate: 그래디언트 업데이트 스텝 크기
            max_iter: 최대 프라이멀-듀얼 반복 횟수
            eps: 수치 안정성을 위한 작은 상수
            alpha: Packet Loss 서로게이트 함수의 감쇠 계수
            min_latency: 최소 허용 지연시간
            convergence_threshold: 수렴 판단 임계값
        """
        self.total_rbs = total_rbs
        self.num_slices = num_slices
        self.weights = weights
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.min_latency = min_latency
        self.convergence_threshold = convergence_threshold
        
        # 듀얼 변수 초기화
        self.lambda_dual = 0.0
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def throughput_surrogate(self, rb: float, se: float) -> float:
        """
        Log-sum-exp 기반 Throughput 서로게이트 함수
        
        Args:
            rb: 할당된 리소스 블록 수
            se: Spectral Efficiency
            
        Returns:
            Throughput 목적함수 값
        """
        return -np.log(np.sum(np.exp(-rb * se)))
        
    def latency_surrogate(self, rb: float, queue_length: float) -> float:
        """
        큐 길이 기반 Latency 서로게이트 함수
        
        Args:
            rb: 할당된 리소스 블록 수
            queue_length: 현재 큐 길이
            
        Returns:
            Latency 목적함수 값
        """
        return np.maximum(queue_length / (rb + self.eps), self.min_latency)
        
    def packet_loss_surrogate(self, rb: float) -> float:
        """
        지수 함수 기반 Packet Loss 서로게이트 함수
        
        Args:
            rb: 할당된 리소스 블록 수
            
        Returns:
            Packet Loss 목적함수 값
        """
        return np.exp(-self.alpha * rb)
        
    def slice_objective(self, rb: float, se: float, queue_length: float, slice_type: str) -> float:
        """
        단일 슬라이스에 대한 결합 목적함수
        
        Args:
            rb: 할당된 리소스 블록 수
            se: Spectral Efficiency
            queue_length: 현재 큐 길이
            slice_type: 슬라이스 타입 (embb, urllc, be)
            
        Returns:
            슬라이스의 결합 목적함수 값
        """
        w_r = self.weights[f"{slice_type}_throughput"]
        w_l = self.weights[f"{slice_type}_latency"] 
        w_p = self.weights[f"{slice_type}_packetloss"]
        
        return (
            w_r * self.throughput_surrogate(rb, se) +
            w_l * self.latency_surrogate(rb, queue_length) +
            w_p * self.packet_loss_surrogate(rb)
        )

    def decompose_dual_problem(self, slice_problems: List[SliceProblem]) -> List[Dict]:
        """
        듀얼 분해를 통한 슬라이스별 하위 문제 생성
        
        Args:
            slice_problems: 슬라이스별 문제 정보
            
        Returns:
            분해된 하위 문제 리스트
        """
        decomposed_problems = []
        for problem in slice_problems:
            slice_problem = {
                'slice_id': problem.slice_id,
                'objective': lambda rb: (
                    self.slice_objective(
                        rb,
                        problem.spectral_efficiency,
                        problem.queue_length,
                        problem.slice_type
                    ) + self.lambda_dual * rb
                )
            }
            decomposed_problems.append(slice_problem)
        return decomposed_problems

    def optimize_slice(self, problem: Dict) -> float:
        """
        단일 슬라이스에 대한 최적화 수행
        
        Args:
            problem: 슬라이스 최적화 문제
            
        Returns:
            최적화된 리소스 블록 할당량
        """
        rb = self.total_rbs / self.num_slices  # 초기값
        
        # 그래디언트 디센트
        for _ in range(self.max_iter):
            gradient = self.compute_gradient(problem['objective'], rb)
            rb -= self.learning_rate * gradient
            rb = np.maximum(rb, 0)  # 비음수 제약
            
        return rb

    def optimize_parallel(self, slice_problems: List[Dict]) -> np.ndarray:
        """
        병렬 처리를 통한 슬라이스별 최적화
        
        Args:
            slice_problems: 슬라이스별 최적화 문제 리스트
            
        Returns:
            최적화된 리소스 블록 할당 배열
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.optimize_slice, slice_problems))
        return np.array(results)

    def update_dual_variable(self, allocations: np.ndarray):
        """
        듀얼 변수 업데이트
        
        Args:
            allocations: 현재 리소스 블록 할당 상태
        """
        violation = np.sum(allocations) - self.total_rbs
        self.lambda_dual += self.learning_rate * violation
        self.lambda_dual = np.maximum(self.lambda_dual, 0)

    def check_convergence(self, prev_allocations: np.ndarray, curr_allocations: np.ndarray) -> bool:
        """
        최적화 수렴 여부 확인
        
        Args:
            prev_allocations: 이전 할당 상태
            curr_allocations: 현재 할당 상태
            
        Returns:
            수렴 여부
        """
        return np.max(np.abs(prev_allocations - curr_allocations)) < self.convergence_threshold

    def project_to_feasible(self, allocations: np.ndarray) -> np.ndarray:
        """
        실현 가능 영역으로 투영
        
        Args:
            allocations: 현재 리소스 블록 할당 상태
            
        Returns:
            실현 가능한 할당 상태
        """
        if np.sum(allocations) > self.total_rbs:
            allocations *= self.total_rbs / np.sum(allocations)
        return np.round(allocations).astype(int)

    def compute_gradient(self, objective_fn, rb: float, delta: float = 1e-6) -> float:
        """
        수치적 그래디언트 계산
        
        Args:
            objective_fn: 목적함수
            rb: 현재 리소스 블록 할당량
            delta: 그래디언트 계산을 위한 작은 변화량
            
        Returns:
            그래디언트 값
        """
        return (objective_fn(rb + delta) - objective_fn(rb - delta)) / (2 * delta)

    def optimize_slice_allocation(
        self,
        slice_types: List[str],
        spectral_efficiencies: np.ndarray,
        queue_lengths: np.ndarray,
        initial_allocation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        슬라이스 자원 할당 최적화 메인 함수
        
        Args:
            slice_types: 슬라이스 타입 리스트
            spectral_efficiencies: 슬라이스별 spectral efficiency 배열
            queue_lengths: 슬라이스별 큐 길이 배열
            initial_allocation: 초기 할당 상태 (선택사항)
            
        Returns:
            최적화된 리소스 블록 할당 배열
        """
        # 슬라이스 문제 초기화
        slice_problems = [
            SliceProblem(i, se, ql, st)
            for i, (se, ql, st) in enumerate(zip(spectral_efficiencies, queue_lengths, slice_types))
        ]
        
        # 초기 할당
        if initial_allocation is None:
            curr_allocation = np.ones(self.num_slices) * (self.total_rbs / self.num_slices)
        else:
            curr_allocation = initial_allocation.copy()
            
        # 듀얼 변수 초기화
        self.lambda_dual = 0.0
        
        # 프라이멀-듀얼 반복
        for iteration in range(self.max_iter):
            prev_allocation = curr_allocation.copy()
            
            # 듀얼 분해
            decomposed_problems = self.decompose_dual_problem(slice_problems)
            
            # 병렬 프라이멀 업데이트
            curr_allocation = self.optimize_parallel(decomposed_problems)
            
            # 듀얼 업데이트
            self.update_dual_variable(curr_allocation)
            
            # 수렴성 체크
            if self.check_convergence(prev_allocation, curr_allocation):
                self.logger.info(f"Converged after {iteration+1} iterations")
                break
                
            # 로깅
            if (iteration + 1) % 10 == 0:
                self.logger.debug(f"Iteration {iteration+1}: allocation = {curr_allocation}")
                
        # 실현 가능 영역으로 투영
        final_allocation = self.project_to_feasible(curr_allocation)
        
        self.logger.info(f"Final allocation: {final_allocation}")
        return final_allocation