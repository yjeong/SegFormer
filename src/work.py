# -*- coding: utf-8 -*-
import json
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import os
import glob
import shutil

# --- 1. 설정 (시각화 관련 옵션) ---
# 컬러맵 및 알파 블렌딩 설정
COLORMAP = 'viridis'
ALPHA = 0.4

# (선택적) 정규화 범위 설정 (cm 단위 기준) - None이면 각 파일별 min/max 사용
MIN_DEPTH_CM = None
MAX_DEPTH_CM = None

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
# *** 라벨 표시 간격 설정 (예: 20개 등고선마다 라벨 1개) ***
CONTOUR_LABEL_INTERVAL = 5

CONTOUR_LABEL_FMT = lambda x: f'{int(x)}cm' # cm 단위 포맷
CONTOUR_LABEL_FONTSIZE = 10
CONTOUR_LABEL_INLINE = True
CONTOUR_LABEL_INLINE_SPACING = 3

# --- 2. 출력 폴더 설정 및 생성 ---
OUTPUT_BASE_DIR = "results"
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, "image")
OUTPUT_JSON_DIR = os.path.join(OUTPUT_BASE_DIR, "json")

try:
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    print(f"결과 저장 폴더 확인/생성 완료: '{OUTPUT_BASE_DIR}'")
except OSError as e:
    print(f"결과 저장 폴더 생성 실패: {e}")
    exit()


# --- 3. 핵심 시각화 로직 함수 ---
def generate_visualization(image_path, json_path):
    """주어진 이미지와 JSON 경로로 시각화 Figure 객체를 생성하여 반환"""
    try:
        # --- 3.1 깊이 데이터 로드 및 처리 ---
        with open(json_path, 'r') as f:
            depth_data = json.load(f)
        width = depth_data['Width']
        height = depth_data['Height']
        depth_array_flat = np.array(depth_data['Depth'], dtype=np.float32)
        if len(depth_array_flat) != width * height:
            print(f"  경고: JSON 데이터 크기 불일치 ({json_path})")
            return None
        depth_map_m = depth_array_flat.reshape((height, width))
        depth_map_cm = depth_map_m * 100
        depth_map_processed_cm = np.nan_to_num(depth_map_cm, nan=np.inf, posinf=np.inf, neginf=0)
        rotated_depth_map_processed_cm = np.rot90(depth_map_processed_cm, k=-1)
        flipped_rotated_depth_map_cm = rotated_depth_map_processed_cm[::-1, :]
        # processed_height, processed_width = flipped_rotated_depth_map_cm.shape # 변수 사용 안되므로 주석처리 가능

        # --- 3.2 정규화 범위 계산 (cm) ---
        valid_depths_cm = depth_map_cm[np.isfinite(depth_map_cm) & (depth_map_cm > 0)]
        if valid_depths_cm.size == 0:
            print(f"  경고: 유효 깊이 값 없음 ({json_path})")
            min_val_cm, max_val_cm = 0.0, 100.0 # 임시 기본값
        else:
            current_min = np.min(valid_depths_cm)
            current_max = np.max(valid_depths_cm)
            min_val_cm = MIN_DEPTH_CM if MIN_DEPTH_CM is not None else current_min
            max_val_cm = MAX_DEPTH_CM if MAX_DEPTH_CM is not None else current_max
            depth_map_processed_cm[depth_map_processed_cm == np.inf] = max_val_cm
            rotated_depth_map_processed_cm[rotated_depth_map_processed_cm == np.inf] = max_val_cm
            flipped_rotated_depth_map_cm[flipped_rotated_depth_map_cm == np.inf] = max_val_cm

        if min_val_cm >= max_val_cm:
             print(f"  경고: 유효하지 않은 깊이 범위 ({json_path}), min={min_val_cm}, max={max_val_cm}")
             max_val_cm = min_val_cm + 100.0
             depth_map_processed_cm = np.clip(depth_map_processed_cm, min_val_cm, max_val_cm)
             rotated_depth_map_processed_cm = np.clip(rotated_depth_map_processed_cm, min_val_cm, max_val_cm)
             flipped_rotated_depth_map_cm = np.clip(flipped_rotated_depth_map_cm, min_val_cm, max_val_cm)

        norm = mcolors.Normalize(vmin=min_val_cm, vmax=max_val_cm)

        # --- 3.3 컬러맵 적용 ---
        clipped_depth_cm = np.clip(depth_map_processed_cm, min_val_cm, max_val_cm)
        if max_val_cm > min_val_cm:
            normalized_depth = norm(clipped_depth_cm)
        else:
            normalized_depth = np.zeros_like(clipped_depth_cm)

        cmap = plt.colormaps[COLORMAP]
        colored_depth_rgba = cmap(normalized_depth)
        colored_depth_rgba_uint8 = (colored_depth_rgba * 255).astype(np.uint8)
        depth_map_image_rgba = Image.fromarray(colored_depth_rgba_uint8, 'RGBA')
        rotated_colored_depth_image_rgba = depth_map_image_rgba.rotate(-90, expand=True)

        # --- 3.4 원본 이미지 로드 및 블렌딩 ---
        original_image = Image.open(image_path).convert('RGBA')
        original_width, original_height = original_image.size
        resized_rotated_colored_depth_map_rgba = rotated_colored_depth_image_rgba.resize(original_image.size, resample=RESIZE_FILTER)
        r, g, b, a_original = resized_rotated_colored_depth_map_rgba.split()
        new_alpha_channel = Image.new('L', resized_rotated_colored_depth_map_rgba.size, color=int(ALPHA * 255))
        foreground_image = Image.merge('RGBA', (r, g, b, new_alpha_channel))
        blended_image = Image.alpha_composite(original_image, foreground_image)

        # --- 3.5 Matplotlib으로 최종 Figure 생성 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(blended_image)

        if ADD_CONTOURS:
            contour_set = ax.contour(flipped_rotated_depth_map_cm,
                                     levels=CONTOUR_LEVELS, # 200개 레벨
                                     colors=CONTOUR_COLOR,
                                     linewidths=CONTOUR_LINEWIDTH,
                                     alpha=CONTOUR_ALPHA,
                                     extent=[0, original_width, original_height, 0])

            if ADD_CONTOUR_LABELS:
                # *** 라벨 표시 레벨 선택 (개수 간격 기준) ***
                calculated_levels = contour_set.levels
                levels_to_label = [] # 라벨링할 레벨 리스트 초기화
                if calculated_levels.size > 0 and CONTOUR_LABEL_INTERVAL > 0:
                    # 간격(INTERVAL)에 맞춰 레벨 선택
                    # 예: [::20] -> 0, 20, 40, ... 번째 레벨 인덱스에 해당하는 값 선택
                    indices = np.arange(len(calculated_levels))
                    label_indices = indices[::CONTOUR_LABEL_INTERVAL]
                    if len(label_indices) > 0:
                        levels_to_label = calculated_levels[label_indices]
                    # 만약 첫번째나 마지막 레벨을 항상 포함하고 싶다면 추가 로직 구현 가능
                elif calculated_levels.size > 0: # 간격이 0이거나 음수면 모든 레벨 표시 (주의: 많을 수 있음)
                     levels_to_label = calculated_levels

                if len(levels_to_label) > 0:
                    ax.clabel(contour_set,
                              levels=levels_to_label, # 계산된 레벨 리스트 사용
                              inline=CONTOUR_LABEL_INLINE,
                              inline_spacing=CONTOUR_LABEL_INLINE_SPACING,
                              fmt=CONTOUR_LABEL_FMT,
                              fontsize=CONTOUR_LABEL_FONTSIZE)
                    #print(f"  정보: 등고선 라벨 추가 완료 (약 {CONTOUR_LABEL_INTERVAL}개 마다)")
                #else:
                #    print(f"  정보: 라벨 표시할 등고선 레벨을 찾지 못함 ({json_path})")


        ax.axis('off')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Depth (cm)')
        file_basename = os.path.basename(os.path.dirname(json_path))
        title = f"Depth Map Overlay - {file_basename} (Alpha: {ALPHA})"
        if ADD_CONTOURS: title += " with Contours"
        if ADD_CONTOUR_LABELS and ADD_CONTOURS: title += " & Labels (cm)"
        plt.suptitle(title, y=0.95, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        return fig

    except FileNotFoundError:
        print(f"  오류: 파일 없음 {image_path} 또는 {json_path}")
        return None
    except Exception as e:
        print(f"  오류: 시각화 생성 중 예외 발생 ({json_path}): {e}")
        plt.close('all')
        return None

# --- 4. 배치 처리 메인 루프 ---
INPUT_BASE_DIR = "/mnt/nas4/lsj/DB-practice/data_second"
input_dirs_found = glob.glob(os.path.join(INPUT_BASE_DIR, "[1-9]*_*"))
input_dirs_sorted = sorted(input_dirs_found, key=lambda x: int(os.path.basename(x).split('_')[0]))

print(f"총 {len(input_dirs_sorted)}개의 입력 폴더를 찾았습니다.")
print("배치 처리를 시작합니다...")

output_count = 1

for input_dir in input_dirs_sorted:
    image_path = os.path.join(input_dir, "image.jpg")
    json_path = os.path.join(input_dir, "depth.json")

    print(f"[{output_count}/{len(input_dirs_sorted)}] 처리 중: {input_dir}")

    if os.path.exists(image_path) and os.path.exists(json_path):
        fig = generate_visualization(image_path, json_path)

        if fig is not None:
            output_num_str = f"{output_count:05d}"
            output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"{output_num_str}.jpg")
            output_json_path = os.path.join(OUTPUT_JSON_DIR, f"{output_num_str}.json")

            try:
                fig.savefig(output_image_path, format='jpg', dpi=150, bbox_inches='tight', pad_inches=0.1)
                print(f"  -> 이미지 저장 완료: {output_image_path}")
                shutil.copyfile(json_path, output_json_path)
                print(f"  -> JSON 복사 완료: {output_json_path}")
                output_count += 1
            except Exception as e:
                print(f"  -> 오류: 결과 저장 중 예외 발생 ({input_dir}): {e}")
            finally:
                plt.close(fig) # 메모리 해제
        else:
            print(f"  -> 건너뛰기 (시각화 함수 실패)")
            plt.close('all')
    else:
        print(f"  -> 건너뛰기 (image.jpg 또는 depth.json 파일을 찾을 수 없음)")

print(f"--- 배치 처리 완료. 총 {output_count - 1}개 쌍 처리됨. ---")