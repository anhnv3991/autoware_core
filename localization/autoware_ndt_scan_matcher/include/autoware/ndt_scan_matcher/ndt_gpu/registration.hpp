#ifndef NDT_GPU_REGISTRATION_HPP_
#define NDT_GPU_REGISTRATION_HPP_

#include "common.hpp"
#include "point_cloud.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace gpu
{

template <typename PointTargetType, typename PointSourceType>
class Registration
{
public:
  using PclTargetType = pcl::PointCloud<PointTargetType>;
  using PclSourceType = pcl::PointCloud<PointSourceType>;
  using PclTargetPtr = typename PclTargetType::SharedPtr;
  using PclSourcePtr = typename PclSourceType::SharedPtr;
  using GpuTargetType = PointCloud<PointTargetType>;
  using GpuSourceType = PointCloud<PointSourceType>;
  using GpuTargetPtr = typename GpuTargetType::SharedPtr;
  using GpuSourcePtr = typename GpuSourceType::SharedPtr;

  Registration()
  {
    max_iterations_ = 0;
    converged_ = false;
    nr_iterations_ = 0;
    trans_eps_ = 0;

    source_cloud_.reset(new GpuSourceType);
    trans_cloud_.reset(new GpuSourceType);
    target_cloud_.reset(new GpuTargetType);
  }

  Registration(const Registration & other)
  : source_cloud_(other.source_cloud_), 
    trans_cloud_(other.trans_cloud_), 
    target_cloud_(other.target_cloud_)
  {
    trans_eps_ = other.trans_eps_;
    max_iterations_ = other.max_iterations_;

    converged_ = other.converged_;

    nr_iterations_ = other.nr_iterations_;
    final_transformation_ = other.final_transformation_;
    transformation_ = other.transformation_;
    previous_transformation_ = other.previous_transformation_;
  }

  Registration(Registration && other)
  : source_cloud_(std::move(other.source_cloud_)), 
    trans_cloud_(std::move(other.trans_cloud_)), 
    target_cloud_(std::move(other.target_cloud_))
  {
    trans_eps_ = other.trans_eps_;
    max_iterations_ = other.max_iterations_;

    converged_ = other.converged_;

    nr_iterations_ = other.nr_iterations_;
    final_transformation_ = other.final_transformation_;
    transformation_ = other.transformation_;
    previous_transformation_ = other.previous_transformation_;
  }


  void align(const Eigen::Matrix<float, 4, 4> & guess)
  {
    converged_ = false;

    final_transformation_ = transformation_ = previous_transformation_ =
      Eigen::Matrix<float, 4, 4>::Identity();

    computeTransformation(guess);
  }

  void setTransformationEpsilon(double trans_eps) { trans_eps_ = trans_eps; }

  double getTransformationEpsilon() const { return trans_eps_; }

  void setMaximumIterations(int max_itr) { max_iterations_ = max_itr; }

  int getMaximumIterations() const { return max_iterations_; }

  Eigen::Matrix<float, 4, 4> getFinalTransformation() const { return final_transformation_; }

  int getFinalNumIteration() const { return nr_iterations_; }

  bool hasConverged() const { return converged_; }

  /* Set input Scanned point cloud.
   * Copy input points from the main memory to the GPU memory */
  void setInputSource(PclSourcePtr input) { *source_cloud_ = *input; }
  void setInputSource(GpuSourcePtr input) { source_cloud_ = input; }

  /* Set input reference map point cloud.
   * Copy input points from the main memory to the GPU memory */
  void setInputTarget(PclTargetPtr input) { *target_cloud_ = *input; }
  void setInputTarget(GpuTargetPtr input) { target_cloud_ = input; }

protected:
  virtual void computeTransformation(const Eigen::Matrix<float, 4, 4> & guess)
  {
    std::cerr << "Error: Unsupported by Registration" << std::endl;
  }

  double trans_eps_;
  int max_iterations_;

  // Original source cloud
  GpuSourcePtr source_cloud_;

  // Transformed point clouds
  GpuSourcePtr trans_cloud_;

  bool converged_;
  int nr_iterations_;

  Eigen::Matrix<float, 4, 4> final_transformation_, transformation_, previous_transformation_;

  // Reference map point
  GpuTargetPtr target_cloud_;
};
}  // namespace gpu

#endif
