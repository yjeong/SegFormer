# -*- coding: utf-8 -*-
import json
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math # 라벨 레벨 계산용

# --- 설정 ---
JSON_FILE_PATH = 'depth.json'
ORIGINAL_IMAGE_PATH = 'image.jpg'
OUTPUT_FIGURE_PATH = 'rotated_blended_contours_cm.png' # 새 출력 파일 이름

# 컬러맵 및 알파 블렌딩 설정
COLORMAP = 'viridis'
ALPHA = 0.4

# (선택적) 정규화 범위 설정 (이제 cm 단위 기준)
# None으로 두면 데이터의 min/max를 자동으로 찾음
MIN_DEPTH_CM = None # 예: 80
MAX_DEPTH_CM = None # 예: 450

# 리사이즈 필터
RESIZE_FILTER = Image.Resampling.BILINEAR

# --- 등고선(Contour) 설정 ---
ADD_CONTOURS = True
# *** 등고선 개수 200으로 설정 ***
CONTOUR_LEVELS = 200
CONTOUR_COLOR = 'white'
CONTOUR_LINEWIDTH = 0.4
CONTOUR_ALPHA = 0.6

# --- 등고선 라벨 설정 (cm 단위 기준) ---
ADD_CONTOUR_LABELS = True
# 라벨 표시할 간격 (cm) - 동적 계산 시 사용
CONTOUR_LABEL_STEP_CM = 5 # 예: 50cm 마다 라벨 표시
# 라벨 형식 (소수점 없이 cm 단위 표시)
CONTOUR_LABEL_FMT = lambda x: f'{int(x)}cm' # 또는 '%.0f cm'
CONTOUR_LABEL_FONTSIZE = 10
CONTOUR_LABEL_INLINE = True
CONTOUR_LABEL_INLINE_SPACING = 3

# --- 1. 깊이 데이터 로드 ---
try:
    with open(JSON_FILE_PATH, 'r') as f:
        depth_data = json.load(f)
    print(f"'{JSON_FILE_PATH}' 로드 완료.")
    width = depth_data['Width']
    height = depth_data['Height']
    depth_array_flat = np.array(depth_data['Depth'], dtype=np.float32)
    if len(depth_array_flat) != width * height: raise ValueError("JSON 데이터 크기 불일치.")
    depth_map_m = depth_array_flat.reshape((height, width)) # 원본 (H, W) - 미터 단위
    print(f"원본 깊이 데이터 ({height}, {width}) 로드 완료 (단위: m).")
except Exception as e: print(f"깊이 데이터 로드/파싱 오류: {e}"); exit()

# --- 2. 단위 변환 (m -> cm) 및 데이터 처리/회전/반전 ---
# *** 미터 단위를 센티미터 단위로 변환 ***
depth_map_cm = depth_map_m * 100

# 이후 모든 처리는 cm 단위 데이터로 수행
depth_map_processed_cm = np.nan_to_num(depth_map_cm, nan=np.inf, posinf=np.inf, neginf=0)
rotated_depth_map_processed_cm = np.rot90(depth_map_processed_cm, k=-1)
flipped_rotated_depth_map_cm = rotated_depth_map_processed_cm[::-1, :]
processed_height, processed_width = flipped_rotated_depth_map_cm.shape
print(f"깊이 데이터 cm 변환, 회전, 반전 완료. 형태: ({processed_height}, {processed_width})")

# --- 3. 정규화 범위 계산 (cm 단위 기준) ---
# cm 데이터에서 유효값 찾기
valid_depths_cm = depth_map_cm[np.isfinite(depth_map_cm) & (depth_map_cm > 0)] # 0보다 큰 유한값
if valid_depths_cm.size == 0:
    print("경고: 유효 깊이 값(cm) 없음. 기본 범위 [0, 100] 사용.")
    min_val_cm, max_val_cm = 0.0, 100.0
else:
    # 설정값(MIN_DEPTH_CM) 또는 실제 최소값 사용
    min_val_cm = MIN_DEPTH_CM if MIN_DEPTH_CM is not None else np.min(valid_depths_cm)
    # 설정값(MAX_DEPTH_CM) 또는 실제 최대값 사용
    max_val_cm = MAX_DEPTH_CM if MAX_DEPTH_CM is not None else np.max(valid_depths_cm)
    # 계산된 max 값으로 Inf 대체 (모든 관련 배열에 적용)
    depth_map_processed_cm[depth_map_processed_cm == np.inf] = max_val_cm
    rotated_depth_map_processed_cm[rotated_depth_map_processed_cm == np.inf] = max_val_cm
    flipped_rotated_depth_map_cm[flipped_rotated_depth_map_cm == np.inf] = max_val_cm


if min_val_cm >= max_val_cm:
    print(f"경고: 유효하지 않은 깊이 범위(cm) (최소={min_val_cm}, 최대={max_val_cm}). 대체 범위 사용.")
    max_val_cm = np.max(depth_map_cm[np.isfinite(depth_map_cm)]) if np.any(np.isfinite(depth_map_cm)) else 100.0
    min_val_cm = 0.0
    if min_val_cm >= max_val_cm: max_val_cm = min_val_cm + 100.0 # 예: 1미터 범위

print(f"정규화 범위(cm): 최소={min_val_cm:.1f}cm, 최대={max_val_cm:.1f}cm")
# 정규화 객체 생성 (cm 단위 기준)
norm = mcolors.Normalize(vmin=min_val_cm, vmax=max_val_cm)

# --- 4. 원본 깊이 데이터(cm) 정규화 (컬러맵 적용용) ---
# cm 단위 처리된 데이터 사용 (회전/반전 안된 것)
clipped_depth_cm = np.clip(depth_map_processed_cm, min_val_cm, max_val_cm)
if max_val_cm > min_val_cm:
    normalized_depth = norm(clipped_depth_cm) # cm 값 -> 0~1 정규화
else:
    normalized_depth = np.zeros_like(clipped_depth_cm)

# --- 5. 컬러맵 적용 및 PIL 이미지 생성 ---
# 정규화된 데이터(0~1)에 컬러맵 적용하는 것은 동일
cmap = plt.colormaps[COLORMAP]
colored_depth_rgba = cmap(normalized_depth)
colored_depth_rgba_uint8 = (colored_depth_rgba * 255).astype(np.uint8)
depth_map_image_rgba = Image.fromarray(colored_depth_rgba_uint8, 'RGBA')
print(f"'{COLORMAP}' 컬러맵 적용 완료.")

# --- 6. 색상 입힌 PIL 이미지 회전 ---
rotated_colored_depth_image_rgba = depth_map_image_rgba.rotate(-90, expand=True)
print("색상 뎁스 맵 PIL 이미지 회전 완료.")

# --- 7. 원본 이미지 로드 ---
try:
    original_image = Image.open(ORIGINAL_IMAGE_PATH).convert('RGBA')
    print(f"원본 이미지 '{ORIGINAL_IMAGE_PATH}' 로드 완료.")
    original_width, original_height = original_image.size
except Exception as e: print(f"원본 이미지 로드 오류: {e}"); exit()

# --- 8. 회전된 컬러 뎁스 맵 리사이즈 ---
resized_rotated_colored_depth_map_rgba = rotated_colored_depth_image_rgba.resize(original_image.size, resample=RESIZE_FILTER)
print(f"회전된 컬러 뎁스 맵 원본 크기 {original_image.size}(으)로 리사이즈 완료.")

# --- 9. 알파 블렌딩 ---
r, g, b, a_original = resized_rotated_colored_depth_map_rgba.split()
new_alpha_channel = Image.new('L', resized_rotated_colored_depth_map_rgba.size, color=int(ALPHA * 255))
foreground_image = Image.merge('RGBA', (r, g, b, new_alpha_channel))
blended_image = Image.alpha_composite(original_image, foreground_image)
print(f"알파 블렌딩 완료 (alpha={ALPHA}).")


# --- 10. Matplotlib으로 최종 결과 생성 (이미지 + 등고선 + 라벨 + 컬러바) ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(blended_image)

if ADD_CONTOURS:
    # 등고선 그리기 (cm 단위 데이터 사용, 200 레벨)
    contour_set = ax.contour(flipped_rotated_depth_map_cm, # cm 단위 데이터
                             levels=CONTOUR_LEVELS, # 200개 레벨
                             colors=CONTOUR_COLOR,
                             linewidths=CONTOUR_LINEWIDTH,
                             alpha=CONTOUR_ALPHA,
                             extent=[0, original_width, original_height, 0])

    # 등고선 라벨 추가 (ADD_CONTOUR_LABELS가 True일 경우)
    if ADD_CONTOUR_LABELS:
        # *** 라벨 표시할 레벨 동적 계산 (cm 단위) ***
        label_step = CONTOUR_LABEL_STEP_CM # 예: 50cm
        # 실제 등고선 레벨 중에서 가장 가까운 값을 찾아서 라벨로 사용
        levels_to_label = []
        for target_level in range(int(min_val_cm), int(max_val_cm) + 1, label_step):
            # 실제 등고선 레벨 중에서 가장 가까운 값을 찾음
            closest_level = min(contour_set.levels, key=lambda x: abs(x - target_level))
            if abs(closest_level - target_level) <= label_step/2:  # 허용 오차 범위 내에 있는 경우만 선택
                levels_to_label.append(closest_level)

        if levels_to_label: # 계산된 라벨 레벨이 있으면
             ax.clabel(contour_set,
                       levels=levels_to_label, # 계산된 cm 레벨 리스트 사용
                       inline=CONTOUR_LABEL_INLINE,
                       inline_spacing=CONTOUR_LABEL_INLINE_SPACING,
                       fmt=CONTOUR_LABEL_FMT, # cm 단위 포맷 적용
                       fontsize=CONTOUR_LABEL_FONTSIZE)
             #print(f"등고선 라벨 추가 완료 (Levels(cm): {levels_to_label})")
        else:
             print("계산된 라벨 표시 레벨이 없습니다. 데이터 범위를 확인하거나 CONTOUR_LABEL_STEP_CM을 조절하세요.")


# 최종 꾸미기
ax.axis('off')

# 컬러바 추가 (norm 객체는 이미 cm 단위 기준)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
# *** 컬러바 라벨 cm 단위로 변경 ***
cbar.set_label('Depth (cm)')

# 영어 제목 설정
title = f"Depth Map Overlay (Alpha: {ALPHA})"
if ADD_CONTOURS: title += " with Contours"
if ADD_CONTOUR_LABELS and ADD_CONTOURS: title += " & Labels (cm)"
plt.suptitle(title, y=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- 11. 결과 저장 및 표시 ---
try:
    plt.savefig(OUTPUT_FIGURE_PATH, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"최종 결과(cm 단위)를 '{OUTPUT_FIGURE_PATH}'(으)로 저장했습니다.")
    plt.show()
    print("결과를 화면에 표시했습니다.")
except Exception as e: print(f"결과 저장/표시 오류: {e}")