#include "candidates.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <numeric>
#include <utility>

namespace tech_core_estimation::detail
{
namespace
{

/** @brief Center the N-by-2 or N-by-3 coordinate matrix for a translation-independent rank test. */
cv::Mat centeredCoordinates(const cv::Mat & points)
{
  const cv::Mat coordinates = points.reshape(1);
  cv::Mat mean;
  cv::reduce(coordinates, mean, 0, cv::REDUCE_AVG);
  return coordinates - cv::repeat(mean, coordinates.rows, 1);
}

/** @brief Singular values in decreasing order for centered landmark coordinates. */
cv::Mat singularValues(const cv::Mat & points)
{
  cv::Mat singular;
  cv::SVD::compute(centeredCoordinates(points), singular);
  return singular;
}

/** @brief Reference relative rank test for both model and image coordinates. */
bool spansPlane(const cv::Mat & singular)
{
  return singular.at<double>(0) > 1e-12 && singular.at<double>(1) > singular.at<double>(0) * 1e-7;
}

/** @brief Whether a noncollinear 3D point set lies in a plane. */
bool planar(const cv::Mat & singular)
{
  return singular.at<double>(2) <= singular.at<double>(0) * 1e-7;
}

/** @brief Append a new support once, preserving the generation order for tied scores. */
void addGroup(std::vector<std::vector<int>> & groups, const std::vector<int> & support)
{
  if (std::find(groups.begin(), groups.end(), support) == groups.end()) {
    groups.push_back(support);
  }
}

/** @brief Screen planar points in undistorted pixels and retain the homography consensus. */
std::vector<int> homographySupport(const Correspondences & observations, const Calibration & camera,
                                   const PnPConfig & config, const cv::Mat & centered,
                                   const cv::Mat & basis)
{
  const cv::Mat plane = centered * basis.rowRange(0, 2).t();
  std::vector<cv::Point2d> undistorted;
  cv::undistortPoints(observations.pixels, undistorted, camera.matrix, camera.distortion,
                      cv::noArray(), camera.matrix);
  std::vector<unsigned char> mask;
  const cv::Mat homography = cv::findHomography(plane, undistorted, cv::RANSAC,
    config.reprojection_threshold, mask, config.ransac_iterations, config.ransac_confidence);
  std::vector<int> support;
  if (!homography.empty() && cv::checkRange(homography)) {
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) {
        support.push_back(static_cast<int>(i));
      }
    }
  }
  return support;
}

/** @brief Append the SQPnP or AP3P RANSAC seed and its support. */
void addRansac(const Correspondences & observations, const Calibration & camera,
               const PnPConfig & config, cv::SolvePnPMethod method,
               std::vector<std::vector<int>> & groups, std::vector<Seed> & seeds)
{
  Seed seed;
  bool solved;
  try {
    solved = cv::solvePnPRansac(observations.points, observations.pixels, camera.matrix,
      camera.distortion, seed.pose.rotation, seed.pose.translation, false, config.ransac_iterations,
      static_cast<float>(config.reprojection_threshold), config.ransac_confidence, seed.support, method);
  } catch (const cv::Exception &) {
    // OpenCV 4.6 SQPnP rejects subpixel clusters during the RANSAC refit.
    return;
  }
  if (solved && seed.support.size() >= 4) {
    std::sort(seed.support.begin(), seed.support.end());
    addGroup(groups, seed.support);
    seeds.push_back(std::move(seed));
  }
}

/** @brief Collect all solutions for a group, including both planar IPPE branches. */
void addGeneric(const Correspondences & group, const Calibration & camera, cv::SolvePnPMethod method,
                const std::vector<int> & support, std::vector<Seed> & seeds)
{
  std::vector<cv::Vec3d> rotations, translations;
  try {
    cv::solvePnPGeneric(group.points, group.pixels, camera.matrix, camera.distortion, rotations,
                        translations, false, method);
  } catch (const cv::Exception &) {
    // A small normalized image variance can also reject the direct SQPnP fit.
    return;
  }
  for (size_t i = 0; i < rotations.size(); ++i) {
    seeds.push_back({{rotations[i], translations[i]}, support});
  }
}

}  // namespace

Correspondences Correspondences::subset(const std::vector<int> & indices) const
{
  Correspondences group;
  group.points.reserve(indices.size());
  group.pixels.reserve(indices.size());
  for (int i : indices) {
    group.points.push_back(points[i]);
    group.pixels.push_back(pixels[i]);
  }
  return group;
}

bool noncollinear(const Correspondences & observations)
{
  return spansPlane(singularValues(cv::Mat(observations.points))) &&
         spansPlane(singularValues(cv::Mat(observations.pixels)));
}

std::vector<Seed> generateSeeds(const Correspondences & observations, const Calibration & camera,
                                const PnPConfig & config)
{
  std::vector<Seed> seeds;
  std::vector<int> all(observations.points.size());
  std::iota(all.begin(), all.end(), 0);
  std::vector<std::vector<int>> groups{all};
  if (all.size() >= 5) {
    const cv::Mat centered = centeredCoordinates(cv::Mat(observations.points));
    cv::Mat singular, basis;
    cv::SVD::compute(centered, singular, cv::noArray(), basis);
    if (planar(singular)) {
      const auto support = homographySupport(observations, camera, config, centered, basis);
      if (support.size() >= 4) {
        addGroup(groups, support);
      }
    } else {
      addRansac(observations, camera, config, cv::SOLVEPNP_SQPNP, groups, seeds);
    }
    addRansac(observations, camera, config, cv::SOLVEPNP_AP3P, groups, seeds);
    if (all.size() <= 6) {
      for (int excluded : all) {
        auto support = all;
        support.erase(support.begin() + excluded);
        addGroup(groups, support);
      }
    }
  }
  for (const auto & support : groups) {
    const auto group = observations.subset(support);
    const cv::Mat singular = singularValues(cv::Mat(group.points));
    if (!spansPlane(singular) || !spansPlane(singularValues(cv::Mat(group.pixels)))) {
      continue;
    }
    addGeneric(group, camera, cv::SOLVEPNP_SQPNP, support, seeds);
    if (planar(singular)) {
      addGeneric(group, camera, cv::SOLVEPNP_IPPE, support, seeds);
    }
  }
  return seeds;
}

}  // namespace tech_core_estimation::detail
