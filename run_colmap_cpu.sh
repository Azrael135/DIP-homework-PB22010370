#!/usr/bin/env bash
set -e

# =========================
# Paths
# =========================
PROJECT_DIR=$(pwd)
IMAGE_PATH=${PROJECT_DIR}/data/images
WORKSPACE=${PROJECT_DIR}/data/colmap_cpu
DATABASE_PATH=${WORKSPACE}/database.db
SPARSE_PATH=${WORKSPACE}/sparse

mkdir -p ${WORKSPACE}
mkdir -p ${SPARSE_PATH}

echo "Project dir: ${PROJECT_DIR}"
echo "Image path : ${IMAGE_PATH}"
echo "Workspace  : ${WORKSPACE}"

# =========================
# 1. Feature Extraction
# CPU mode, because this computer has no GPU.
# Rendered images usually share the same virtual camera,
# so single_camera=1 is reasonable.
# =========================
colmap feature_extractor \
    --database_path ${DATABASE_PATH} \
    --image_path ${IMAGE_PATH} \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --SiftExtraction.use_gpu 0

# =========================
# 2. Feature Matching
# 50 images only, exhaustive matching is acceptable.
# =========================
colmap exhaustive_matcher \
    --database_path ${DATABASE_PATH} \
    --SiftMatching.use_gpu 0

# =========================
# 3. Sparse Reconstruction / Mapper
# This includes COLMAP's internal bundle adjustment.
# =========================
colmap mapper \
    --database_path ${DATABASE_PATH} \
    --image_path ${IMAGE_PATH} \
    --output_path ${SPARSE_PATH}

# =========================
# 4. Analyze reconstruction
# Usually the main model is sparse/0.
# =========================
colmap model_analyzer \
    --path ${SPARSE_PATH}/0

# =========================
# 5. Export sparse model to PLY for MeshLab
# =========================
colmap model_converter \
    --input_path ${SPARSE_PATH}/0 \
    --output_path ${WORKSPACE}/sparse_points.ply \
    --output_type PLY

echo "Done."
echo "Sparse model folder: ${SPARSE_PATH}/0"
echo "Sparse point cloud : ${WORKSPACE}/sparse_points.ply"
