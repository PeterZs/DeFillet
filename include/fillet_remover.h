//
// Created by xiaowuga on 2025/9/28.
//

#ifndef FILLET_REMOVER_H
#define FILLET_REMOVER_H

#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/point_cloud.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>


using namespace easy3d;
using namespace std;

namespace DeFillet {

    class FilletRemoverParameters {

    public:
        std::string input_path;
        std::string label_path;
        std::string out_dir;
        float beta_e, beta_f, beta_c;
        float angle_thr;
        int num_opt_iter;

        int num_threads;
    };

    class FilletRemover {
    public:
        FilletRemover();
        void initialize(easy3d::SurfaceMesh* mesh, FilletRemoverParameters& parameters, const std::vector<int>& fillet_labels);
        void optimize();
    public:
        easy3d::SurfaceMesh* mesh_;
        easy3d::SurfaceMesh* defillet_mesh_;
        easy3d::SurfaceMesh* focus_area_;
        easy3d::SurfaceMesh* fillet_mesh_;
        easy3d::SurfaceMesh* non_fillet_mesh_;
        FilletRemoverParameters parameters_;
        std::vector<std::pair<easy3d::vec3, easy3d::vec3>> tar_normals;


        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
        Eigen::VectorXd d_;

    };
}



#endif //FILLET_REMOVER_H
