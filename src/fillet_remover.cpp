//
// Created by xiaowuga on 2025/9/28.
//

#include "fillet_remover.h"
#include <easy3d/util/stop_watch.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/algo/surface_mesh_geometry.h>


namespace DeFillet {
    FilletRemover::FilletRemover(FilletRemoverParameters& parameters) {
        parameters_ = parameters;
    }

    void FilletRemover::initialize(easy3d::SurfaceMesh* mesh, const std::vector<int>& fillet_labels) {

        std::cout << "Start removal initialization..." <<std::endl;
        easy3d::StopWatch sw; sw.start();
        mesh_ = new SurfaceMesh(*mesh);
        auto labels = mesh_->face_property<int>("f:labels");

    }

}