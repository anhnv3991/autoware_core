#ifndef NDT_GPU_VOXEL_GRID_COVARIANCE_HPP_
#define NDT_GPU_VOXEL_GRID_COVARIANCE_HPP_

#include <thrust/device_vector.h>

#include <memory>

#include "debug.hpp"
#include "common.hpp"
#include "matrix.hpp"
#include "point_cloud.hpp"

namespace gpu
{

template <typename PointT>
class VoxelGridCovariance
{
    using PclCloudType = pcl::PointCloud<PointT>;
    using PclCloudPtr = typename PclCloudType::SharedPtr;
    using GpuCloudType = PointCloud<PointT>;
    using GpuCloudPtr = typename GpuCloudType::SharedPtr;

public:
    VoxelGridCovariance();
    VoxelGridCovariance(const VoxelGridCovariance & other);
    VoxelGridCovariance(VoxelGridCovariance && other);
    VoxelGridCovariance& operator=(const VoxelGridCovariance & other);
    VoxelGridCovariance& operator=(VoxelGridCovariance && other);

    void setLeafSize(Eigen::Vector3f & leaf_size) { leaf_size_ = leaf_size; }
    // No need to keep a copy of the cloud
    void setInputCloud(const PclCloudType & input_cloud);
    void setInputCloud(const GpuCloudType & input_cloud);

    void applyFilter(const GpuCloudType & input_cloud);    
    
private:

    Eigen::Vector3f leaf_size_;
};
}

#endif