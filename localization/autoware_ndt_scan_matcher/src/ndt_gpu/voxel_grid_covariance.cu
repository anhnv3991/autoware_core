#include <autoware/ndt_scan_matcher/ndt_gpu/voxel_grid_covariance.hpp>

namespace gpu
{

template <typename PointT>
void VoxelGridCovariance<PointT>::applyFilter(const GpuCloudType & input)
{
    // Find the min/max coordinates 
    // Allocate memory for voxels
    // Distribute points to voxels
    // Compute centroids and covariances and inverse covariances
}

}