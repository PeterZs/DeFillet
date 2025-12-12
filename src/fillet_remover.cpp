//
// Created by xiaowuga on 2025/9/28.
//

#include "fillet_remover.h"
#include <easy3d/util/stop_watch.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/algo/surface_mesh_geometry.h>

#include <surafce_mesh_segmenter.h>
#include <utils.h>

namespace DeFillet {
    FilletRemover::FilletRemover(FilletRemoverParameters& parameters) {
        parameters_ = parameters;
    }

    void FilletRemover::initialize(easy3d::SurfaceMesh* mesh, const std::vector<int>& fillet_labels) {

        std::cout << "Start removal initialization..." <<std::endl;
        easy3d::StopWatch sw; sw.start();
        mesh_ = new SurfaceMesh(*mesh);
        auto labels = mesh_->face_property<int>("f:labels");
        for(auto f : mesh_->faces()) {
            labels[f] = 0;
        }
        for(auto e : mesh_->edges()) {
            auto f0 = mesh_->face(e, 0);
            auto f1 = mesh_->face(e, 1);
            int idx0 = f0.idx(), idx1 = f1.idx();
            if(idx0 != -1 && idx1 != -1 &&  fillet_labels[idx0] + fillet_labels[idx1] < 2) {
                labels[f0] = 1; labels[f1] = 1;
            }
        }

        easy3d::SurfaceMeshSegmenter sms(mesh_);
        focus_area_ =  sms.segment<int>(labels, 1);


        auto idx = focus_area_->face_property<int>("f:original_index"); //original face index
        std::map<int, easy3d::vec3> fixed;
        double angle_thr = parameters_.angle_thr;
        for(auto v : focus_area_->vertices()) {
            int ct = 0, flag = 0;
            for(auto h : focus_area_->halfedges(v)) {
                auto oh = focus_area_->opposite(h);
                auto f0 = focus_area_->face(h);
                auto f1 = focus_area_->face(oh);

                if(f0.is_valid()) {
                    int id = idx[f0];
                    auto ff = easy3d::SurfaceMesh::Face(id);
                    if(mesh_->is_border(ff)) {
                        flag =1; break;
                    }
                }
                if(f1.is_valid()) {
                    int id = idx[f1];
                    auto ff = easy3d::SurfaceMesh::Face(id);
                    if(mesh_->is_border(ff)) {
                        flag =1; break;
                    }
                }

                if(!f0.is_valid() || !f1.is_valid()) {
                    ct++;
                }
                else if(f0.is_valid() && f1.is_valid()) {
                    easy3d::vec3 n0 = focus_area_->compute_face_normal(f0);
                    easy3d::vec3 n1 = focus_area_->compute_face_normal(f1);
                    double angle = angle_between(n0, n1);
                    if(angle > angle_thr) {
                        flag = 1;
                    }
                }
            }
            if(ct == 2 && flag == 0) {
                fixed[v.idx()] = focus_area_->compute_vertex_normal(v);
            }
        }




    }

}