"""wide_fov_supervision_v2 패키지.

이 패키지는 fisheye source 영상에서 ray 간격이 커지는 영역을 찾아 query ray를
추가하고, Depth Anything V2/DSINE teacher와 ray-aware refiner를 연결해 depth,
normal, BEV 산출물을 만드는 실험 코드이다.
"""

__all__ = ["config", "pipeline"]
