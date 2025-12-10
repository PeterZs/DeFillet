//
// Created by xiaowuga on 2025/8/15.
//

#ifndef UTILS_H
#define UTILS_H

#include <easy3d/core/point_cloud.h>
#include <easy3d/core/surface_mesh.h>

#include <fillet_detector.h>


namespace DeFillet {


    FilletDetectorParameters load_detector_config(const std::string &filename);

    void save_detector_config(const FilletDetectorParameters &params, const std::string &filename);

    void normalize_model(easy3d::SurfaceMesh* mesh, easy3d::vec3& centroids, double& scale);

    void inverse_normalize_model(easy3d::SurfaceMesh* model, easy3d::vec3& centroids, double& scale);

    std::string get_time_stamp(bool for_filename = false);

    /**
    * @brief Compute the (unsigned) angle between two 3D vectors in degrees.
    *
    * Uses atan2(||n1 × n2||, n1 · n2), which is generally more numerically stable
    * than acos of the normalized dot product.
    *
    * @param n1 First vector (e.g., a face normal).
    * @param n2 Second vector (e.g., a face normal).
    * @return Angle in degrees, in the range [0, 180].
    */
    float angle_between(const easy3d::vec3& n1, const easy3d::vec3& n2);

    double gaussian_kernel(double distance, double kernel_bandwidth);


    void sor(const std::vector<easy3d::vec4>& points
                , std::vector<bool>& labels
                , int nb_neighbors
                , int num_sor_iter
                , float std_ratio);

    easy3d::vec3 axis_direction(std::vector<easy3d::vec4> points);

    easy3d::vec3 project_to_line(easy3d::vec3 pos, easy3d::vec3 point, easy3d::vec3 dir);

    void save_components(const easy3d::SurfaceMesh* mesh,
                                const std::vector<easy3d::SurfaceMesh*>components,
                                const std::string path);


    easy3d::SurfaceMesh* split_component(const easy3d::SurfaceMesh* mesh,
                         easy3d::SurfaceMesh::FaceProperty<int>& component_labels,
                         int label);

    void save_field(const easy3d::SurfaceMesh* mesh,
                         const std::vector<float>& field,
                         const std::string path);

    void save_fillet_segmentation(const easy3d::SurfaceMesh* mesh,
                             const std::vector<int>& fillet_label,
                             const std::string path);


    void save_fillet_labels(std::vector<int>& labels, const std::string& path);



}

#endif //UTILS_H
