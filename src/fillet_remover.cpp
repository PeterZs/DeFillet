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

#include <Xin_Wang.h>

#include <igl/cat.h>

namespace DeFillet {
    FilletRemover::FilletRemover() {

    }

    void FilletRemover::initialize(easy3d::SurfaceMesh* mesh, FilletRemoverParameters& parameters, const std::vector<int>& fillet_labels) {
        parameters_ = parameters;
        std::cout << "Start removal initialization..." <<std::endl;
        easy3d::StopWatch sw; sw.start();
        mesh_ = new SurfaceMesh(*mesh);
        auto labels = mesh_->face_property<int>("f:labels");
        easy3d::SurfaceMeshSegmenter sms(mesh_);
        for(auto f : mesh_->faces()) {
            labels[f] = fillet_labels[f.idx()];
        }
        fillet_mesh_ = sms.segment<int>(labels, 1);
        non_fillet_mesh_ = sms.segment<int>(labels, 0);

        for(auto f : mesh_->faces()) {
            labels[f] = 1;
        }

        for(auto e : mesh_->edges()) {
            auto f0 = mesh_->face(e, 0);
            auto f1 = mesh_->face(e, 1);
            int idx0 = f0.idx(), idx1 = f1.idx();
            if(idx0 != -1 && idx1 != -1 &&  fillet_labels[idx0] + fillet_labels[idx1] < 2) {
                labels[f0] = 0; labels[f1] = 0;
            }
        }


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

        int nb_points = focus_area_->n_vertices();
        int nb_faces = focus_area_->n_faces();

        std::vector<CPoint3D> xin_points;
        for(auto v : focus_area_->vertices()) {
            auto& p = focus_area_->position(v);
            xin_points.emplace_back(CPoint3D(p.x, p.y, p.z));
        }
        std::vector<CBaseModel::CFace> xin_faces;

        for(auto f : focus_area_->faces()) {
            int num = 0;
            easy3d::vec3 center = easy3d::vec3(0,0,0);
            std::vector<size_t> indices;
            for(auto v : focus_area_->vertices(f)) {
                center += focus_area_->position(v);
                indices.emplace_back(v.idx());
                num++;
            }
            center /= num;
            xin_points.emplace_back(CPoint3D(center.x, center.y, center.z));

            for(int i = 0; i < num; i++) {
                xin_faces.emplace_back(CBaseModel::CFace(indices[i],
                                                         indices[(i + 1) % num], nb_points + f.idx()));
            }
        }

        CRichModel xin_mesh(xin_points, xin_faces);

        std::set<int> special;
        std::set<int> xin_sources;

        for(auto e : focus_area_->edges()) {
            auto f0 = focus_area_->face(e, 0);
            auto f1 = focus_area_->face(e, 1);
            if(f0.is_valid() && f1.is_valid()) {
                easy3d::vec3 n0 = focus_area_->compute_face_normal(f0);
                easy3d::vec3 n1 = focus_area_->compute_face_normal(f1);
                double angle = angle_between(n0, n1);
                if(angle > angle_thr) {
                    auto f = easy3d::SurfaceMesh::Face(idx[f0]);
                    if(fillet_labels[f.idx()] == 1) {
                        f = f0;
                    }
                    else {
                        f = f1;
                    }
                    for(auto v : focus_area_->vertices(f)) {
                        if(fixed.find(v.idx()) != fixed.end()) {
                            special.insert(v.idx());
                        }
                    }
                }
            }
        }
        for(auto id : fixed) {
            if(special.find(id.first) == special.end()) {
                xin_sources.insert(id.first);
            }
        }

        auto fr = focus_area_->face_property<int>("f:face_root"); // face root
        auto ftn = focus_area_->face_property<easy3d::vec3>("f:tar_normals"); // face target normal
        auto pgd = focus_area_->vertex_property<float>("v:geo_dis");

        CXin_Wang alg(xin_mesh, xin_sources);
        alg.Execute();
        for(int i = 0; i < nb_points; i++) {
            easy3d::SurfaceMesh::Vertex v(i);
            pgd[v] = alg.GetDistanceField()[i];
        }
        std::vector<float> geo;
        for(int i = 0; i < nb_faces; i++) {
            easy3d::SurfaceMesh::Face f(i );
            fr[f] = alg.GetAncestor(i + nb_points);
            ftn[f] = fixed[fr[f]];

        }
        std::vector<easy3d::vec3> sp;
        for(auto id : special) {
            auto v = easy3d::SurfaceMesh::Vertex(id);
            pgd[v] = 0.0;
            sp.emplace_back(focus_area_->position(v));
            easy3d::vec3 n = focus_area_->compute_vertex_normal(v);
            for(auto f : focus_area_->faces(v)) {
                fr[f] = v.idx();
                ftn[f] = n;
            }
        }

        std::vector<easy3d::vec3> s;
        for(auto id : xin_sources) {
            auto v = easy3d::SurfaceMesh::Vertex(id);
            if(special.find(id) == special.end()) {
                s.emplace_back(focus_area_->position(v));
            }
        }


        std::vector<bool>vis(nb_points, false);
        bool state = true;
        int num;
        do {
            num = 0;
            for(auto v : focus_area_->vertices()) {
                if(vis[v.idx()] == state) continue;
                std::queue<easy3d::SurfaceMesh::Vertex>que;
                que.push(v);
                while(!que.empty()) {
                    auto cur = que.front(); que.pop();
                    auto st_h = focus_area_->out_halfedge(cur);
                    auto it = st_h;
                    do {
                        auto cur_f = focus_area_->face(it);
                        auto prev_f = focus_area_->face(focus_area_->prev_around_source(it));
                        auto nxt_f = focus_area_->face(focus_area_->next_around_source(it));
                        if (cur_f.is_valid() && prev_f.is_valid() && nxt_f.is_valid()) {
                            auto n1 = ftn[cur_f];
                            auto n2 = ftn[prev_f];
                            auto n3 = ftn[nxt_f];
                            if (angle_between(n1, n2) > angle_thr && angle_between(n1, n3) > angle_thr
                                && angle_between(n2, n3) < angle_thr) {
                                ftn[cur_f] = ftn[prev_f];
                                fr[cur_f] = fr[prev_f];
                                num++;
                            }
                        }
                        auto tar_v = focus_area_->target(it);
                        if(vis[tar_v.idx()] != state) {
                            que.push(tar_v);
                            vis[tar_v.idx()] = state;
                        }
                        it = focus_area_->next_around_source(it);
                    } while (it != st_h);
                }
            }
            std::cout << "number of faca normals flip = " << num <<std::endl;
        } while(num != 0);


        float len = focus_area_->bounding_box().diagonal_length() * 0.02;

        for (auto f : focus_area_->faces()) {

            easy3d::vec3 normal = ftn[f];


            easy3d::vec3 center(0, 0, 0);
            int v_count = 0;
            for (auto v : focus_area_->vertices(f)) {
                center += focus_area_->position(v);
                v_count++;
            }
            if (v_count > 0) center /= (float)v_count;


            easy3d::vec3 end = center + normal * len;

            tar_normals.emplace_back(std::make_pair(center, end));

        }

        float init_time = sw.elapsed_seconds(5);
        std::cout << "Defillet initialize sussessfully! time="<< init_time <<std::endl;
    }




    void FilletRemover::optimize() {
        easy3d::StopWatch sw; sw.start();
        int nb_points = focus_area_->n_vertices();
        int nb_faces = focus_area_->n_faces();
        std::vector<float> nx(nb_faces);
        std::vector<float> ny(nb_faces);
        std::vector<float> nz(nb_faces);
        auto fr = focus_area_->face_property<int>("f:face_root");
        auto ftn = focus_area_->face_property<easy3d::vec3>("f:tar_normals");
    #pragma omp parallel for
        for(int i = 0; i < nb_faces; i++) {
            auto f = easy3d::SurfaceMesh::Face(i);
            nx[i] = ftn[f].x;
            ny[i] = ftn[f].y;
            nz[i] = ftn[f].z;
        }
        double beta_e = parameters_.beta_e;
        std::vector<Eigen::Triplet<double>> triplets;
        int row = 0;
        for(auto e : focus_area_->edges()) {
            int v0 = focus_area_->vertex(e, 0).idx();
            int v1 = focus_area_->vertex(e, 1).idx();
            auto f0 = focus_area_->face(e, 0);
            auto f1 = focus_area_->face(e, 1);

            if(f0.is_valid() && f1.is_valid()) {
                triplets.emplace_back(Eigen::Triplet<double>(row, v0, beta_e * nx[f0.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v0 + nb_points, beta_e *ny[f0.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v0 + 2 * nb_points, beta_e *nz[f0.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1, beta_e *(-nx[f0.idx()])));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1 + nb_points, beta_e *(-ny[f0.idx()])));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1 + 2 * nb_points, beta_e *(-nz[f0.idx()])));

                row++;

                triplets.emplace_back(Eigen::Triplet<double>(row, v0, beta_e * nx[f1.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v0 + nb_points, beta_e * ny[f1.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v0 + 2 * nb_points, beta_e * nz[f1.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1, beta_e * (-nx[f1.idx()])));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1 + nb_points, beta_e * (-ny[f1.idx()])));
                triplets.emplace_back(Eigen::Triplet<double>(row, v1 + 2 * nb_points, beta_e * (-nz[f1.idx()])));

                row++;
            }
        }
        Eigen::SparseMatrix<double> E(row, nb_points * 3); // Edge Directional Constraint Energy
        E.setFromTriplets(triplets.begin(), triplets.end());

        triplets.clear();
        std::set<int> fixed;
        float beta_f = parameters_.beta_f;

        for(auto f : focus_area_->faces()) {
            int num = 0;
            for(auto v : focus_area_->vertices(f)) {
                triplets.emplace_back(Eigen::Triplet<double>(f.idx(), v.idx(), beta_f * nx[f.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(f.idx(), v.idx() + nb_points, beta_f * ny[f.idx()]));
                triplets.emplace_back(Eigen::Triplet<double>(f.idx(), v.idx() + 2 * nb_points, beta_f * nz[f.idx()]));
                num++;
            }

            triplets.emplace_back(Eigen::Triplet<double>(f.idx(), fr[f], -nx[f.idx()] * beta_f * num));
            triplets.emplace_back(Eigen::Triplet<double>(f.idx(), fr[f] + nb_points, -ny[f.idx()] * beta_f * num));
            triplets.emplace_back(Eigen::Triplet<double>(f.idx(), fr[f] + 2 * nb_points, -nz[f.idx()] * beta_f * num));
            fixed.insert(fr[f]);
        }
        Eigen::SparseMatrix<double> F(nb_faces, nb_points * 3); // Face Normal Constraint Energy
        F.setFromTriplets(triplets.begin(), triplets.end());

        row = 0;
        d_.resize(nb_points * 3);
        triplets.clear();
        std::vector<easy3d::vec3> ff;
        for(auto id : fixed) {
            easy3d::SurfaceMesh::Vertex v(id);
            // ff.emplace_back(focus_area_->position(v));
            auto pos = focus_area_->position(v);
            triplets.emplace_back(Eigen::Triplet<double>(row, v.idx() , 1.0));
            d_[row++] = pos.x;
            triplets.emplace_back(Eigen::Triplet<double>(row, v.idx() + nb_points, 1.0));
            d_[row++] = pos.y;
            triplets.emplace_back(Eigen::Triplet<double>(row,  v.idx() + nb_points * 2, 1.0));
            d_[row++] = pos.z;
        }

        d_.conservativeResize(row);

        Eigen::SparseMatrix<double> D(row, nb_points * 3);
        D.setFromTriplets(triplets.begin(), triplets.end());

        float beta_c = 2 * parameters_.beta_c;
        Eigen::SparseMatrix<double> A;
        igl::cat(1, E, F, A);
        Eigen::SparseMatrix<double> AT = A.transpose();
        Eigen::SparseMatrix<double> ATA = AT * A;
        Eigen::SparseMatrix<double> I(ATA.rows(), ATA.cols()); I.setIdentity();

        Eigen::SparseMatrix<double> Q = ATA + (beta_c * I);
        Eigen::SparseMatrix<double> zero(row, row);
        Eigen::SparseMatrix<double> DT = D.transpose();
        Eigen::SparseMatrix<double> tempMat1;
        Eigen::SparseMatrix<double> tempMat2;
        Eigen::SparseMatrix<double> M;
        igl::cat(1, Q, D, tempMat1);
        igl::cat(1, DT, zero, tempMat2);
        igl::cat(2, tempMat1, tempMat2, M);
        solver_.compute(M);
        if(solver_.info()!= Eigen::Success) {
            std::cout << "decomposition failed" << std::endl;
            return;
        }

        int  num_opt_iter = parameters_.num_opt_iter;
        for(int iter = 0; iter < num_opt_iter; iter++) {
            std::cout << "iter " << iter + 1 << " is processing." << std::endl;
            auto& points = focus_area_->points();

            Eigen::VectorXd p(nb_points * 3);
            for(int i = 0; i < nb_points; i++) {
                p[i] = points[i].x;
                p[i + nb_points] = points[i].y;
                p[i + 2 * nb_points] = points[i].z;
            }

            Eigen::VectorXd b(nb_points * 3 + d_.size());
            b << beta_c * p, d_;
            Eigen::VectorXd x = solver_.solve(b);
            if(solver_.info()!= Eigen::Success) {
                // solving failed
                std::cout << "solving failed" << std::endl;
                return;
            }
            double ec = 0;
            for(auto v : focus_area_->vertices()) {

                int id = v.idx();
                easy3d::vec3 new_p = easy3d::vec3(x[id],
                                                  x[id + nb_points], x[id + 2 * nb_points]);

                focus_area_->position(v) = new_p;
            }

        }

        float opt_time = sw.elapsed_seconds(5);
        std::cout << "Defillet optimize sussessfully! time="<< opt_time <<std::endl;
        // easy3d::SurfaceMeshIO::save("../out/result.ply", focus_area_);
        defillet_mesh_ = new easy3d::SurfaceMesh(*mesh_);
        auto opi = focus_area_->vertex_property<int>("v:original_index");
        for(auto v : focus_area_->vertices()) {
            int id = opi[v];
            easy3d::SurfaceMesh::Vertex vv(id);
            defillet_mesh_->position(vv) = focus_area_->position(v);
        }
    }

}