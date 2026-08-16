#ifndef NDT_GPU_NDT_GPU_HPP_
#define NDT_GPU_NDT_GPU_HPP_

#include <Eigen/Geometry>
#include "registration.hpp"
#include "voxel_grid.hpp"
#include "common.hpp"
#include "point_cloud.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace gpu
{

template <typename PointTargetType, typename PointSourceType>
class NormalDistributionsTransform : public Registration<PointTargetType, PointSourceType>
{
  using BaseRegType = Registration<PointTargetType, PointSourceType>;
  using typename BaseRegType::PclTargetType;
  using typename BaseRegType::PclSourceType;
  using typename BaseRegType::PclTargetPtr;
  using typename BaseRegType::PclSourcePtr;
  using typename BaseRegType::GpuTargetType;
  using typename BaseRegType::GpuSourceType;
  using typename BaseRegType::GpuTargetPtr;
  using typename BaseRegType::GpuSourcePtr;

public:
  NormalDistributionsTransform();
  NormalDistributionsTransform(const NormalDistributionsTransform & other);
  NormalDistributionsTransform(NormalDistributionsTransform && other);

  void setStepSize(double step_size) { step_size_ = step_size; }
  void setResolution(float resolution) { resolution_ = resolution; }
  void setOutlierRatio(double olr) { outlier_ratio_ = olr; }
  double getStepSize() const { return step_size_; }
  float getResolution() const { return resolution_; }
  double getOutlierRatio() const { return outlier_ratio_; }
  double getTransformationProbability() const { return trans_probability_; }
  int getRealIterations() { return real_iterations_; }

  /* Set the input map points */
  void setInputTarget(PclTargetPtr input);
  void setInputTarget(GpuTargetPtr input);

  /* Compute and get fitness score */
  double getFitnessScore(double max_range = DBL_MAX);

  ~NormalDistributionsTransform();

protected:
  void computeTransformation(const Eigen::Matrix<float, 4, 4> & guess);
  double computeDerivatives(
    Eigen::Matrix<double, 6, 1> & score_gradient, 
    Eigen::Matrix<double, 6, 6> & hessian,
    GpuSourceType & trans_cloud,
    Eigen::Matrix<double, 6, 1> pose, 
    bool compute_hessian = true);

private:
  // Copied from ndt.h
  double auxilaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4);

  // Copied from ndt.h
  double auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4);

  double updateIntervalMT(
    double & a_l, double & f_l, double & g_l, double & a_u, double & f_u, double & g_u, double a_t,
    double f_t, double g_t);

  double trialValueSelectionMT(
    double a_l, double f_l, double g_l, double a_u, double f_u, double g_u, double a_t, double f_t,
    double g_t);

  void transformPointCloud(
    GpuSourceType & in_cloud, 
    GpuSourceType & out_cloud,
    Eigen::Matrix<float, 4, 4> transform
  );

  void computeAngleDerivatives(MatrixHost pose, bool compute_hessian = true);

  double computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1> & x, 
    Eigen::Matrix<double, 6, 1> & step_dir, 
    double step_init, double step_max, double step_min, double & score, 
    Eigen::Matrix<double, 6, 1> & score_gradient,
    Eigen::Matrix<double, 6, 6> & hessian, 
    GpuSourceType & out_cloud
  );

  void computeHessian(
    Eigen::Matrix<double, 6, 6> & hessian, float * trans_x, float * trans_y, float * trans_z,
    int points_num, Eigen::Matrix<double, 6, 1> & p);

  double gauss_d1_, gauss_d2_;
  double outlier_ratio_;
  // MatrixHost j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;
  MatrixHost j_ang_;

  // MatrixHost h_ang_a2_, h_ang_a3_, h_ang_b2_, h_ang_b3_, h_ang_c2_, h_ang_c3_, h_ang_d1_,
  // h_ang_d2_, h_ang_d3_,
  //       h_ang_e1_, h_ang_e2_, h_ang_e3_, h_ang_f1_, h_ang_f2_, h_ang_f3_;
  MatrixHost h_ang_;

  // MatrixDevice dj_ang_a_, dj_ang_b_, dj_ang_c_, dj_ang_d_, dj_ang_e_, dj_ang_f_, dj_ang_g_,
  // dj_ang_h_;
  MatrixDevice dj_ang_;

  // MatrixDevice dh_ang_a2_, dh_ang_a3_, dh_ang_b2_, dh_ang_b3_, dh_ang_c2_, dh_ang_c3_,
  // dh_ang_d1_, dh_ang_d2_, dh_ang_d3_,
  //       dh_ang_e1_, dh_ang_e2_, dh_ang_e3_, dh_ang_f1_, dh_ang_f2_, dh_ang_f3_;
  MatrixDevice dh_ang_;

  double step_size_;
  float resolution_;
  double trans_probability_;

  int real_iterations_;

  VoxelGrid voxel_grid_;
};
}  // namespace gpu

#endif
