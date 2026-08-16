#ifndef NDT_GPU_POINT_CLOUD_HPP_
#define NDT_GPU_POINT_CLOUD_HPP_

#include <thrust/device_vector.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <memory>

#include <autoware/ndt_scan_matcher/ndt_gpu/matrix.hpp>

namespace gpu
{

// A pointer used on CUDA kernels
template <typename PointT> class DevCloudPtr;
template <typename PointT> class PointCloud;

// For now, only use pcl::PointXYZ
template <>
class PointCloud<pcl::PointXYZ>
{
public:
    PointCloud();
    PointCloud(const PointCloud& other);
    PointCloud(PointCloud&& other);
    PointCloud<pcl::PointXYZ>& operator=(const PointCloud<pcl::PointXYZ> & other);
    PointCloud<pcl::PointXYZ>& operator=(PointCloud<pcl::PointXYZ> && other);

    // Copy from a pcl::PointCloud
    PointCloud<pcl::PointXYZ>& operator=(const pcl::PointCloud<pcl::PointXYZ> & other);

    // Get a pointer to use on CUDA kernels
    DevCloudPtr data();

    size_t size() const { return point_num_; }
    bool empty() const { return point_num_ == 0; }
    pcl::PointXYZ operator()(int idx) const { return dev_cloud_[idx]; }

    void resize(size_t new_size) { dev_cloud_.resize(new_size); }
    void clear() { dev_cloud_.clear(); }

private:
    thrust::device_vector<float> dev_cloud_;
    size_t point_num_;
};

class DevCloudPtr
{
public:
    DevCloudPtr(float * data, size_t point_num)
    : data_(data), point_num_(point_num)
    {}
    DevCloudPtr(const DevCloudPtr & other);
    DevCloudPtr(DevCloudPtr && other);
    DevCloudPtr& operator=(const DevCloudPtr & other);
    DevCloudPtr& operator=(DevCloudPtr && other);

    // Return a point at index i as a 3x1 column vector
    Point operator[](int idx)
    {
        return Point(data_[idx], data_[idx + point_num_], data_[idx + point_num_ * 2]);
    }
private:
    float * data_;
    size_t point_num_;
};
 
DevCloudPtr PointCloud<pcl::PointXYZ>::data()
{
    return DevCloudPtr(thrust::raw_pointer_cast(dev_cloud_.data()), point_num_);
}

}  // namespace gpu

namespace std
{

template<>
class numeric_limits<gpu::Point>
{
public:
    static CUDAH lowest() noexcept 
    {
        float val = std::numeric_limits<float>::lowest();

        return gpu::Point(val, val, val);
    }

    static CUDAH max() noexcept
    {
        float val = std::numeric_limits<float>::max();

        return gpu::Point(val, val, val);
    }

    static CUDAH min() noexcept
    {
        float val = std::numeric_limits<float>::min();

        return gpu::Point(val, val, val);
    }
private:
}

}

#endif