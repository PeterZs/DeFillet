//
// Created by xiaowuga on 2025/8/16.
//

#include <random>

#include <fillet_detector.h>
#include <voronoi3d.h>
#include <utils.h>
// #include <knn4d.h>
#include <gcp.h>

#include <easy3d/util/stop_watch.h>
#include <easy3d/algo/surface_mesh_geometry.h>
#include <easy3d/fileio/point_cloud_io.h>


#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/hessian_energy.h>
#include <igl/curved_hessian_energy.h>
#include <igl/avg_edge_length.h>


#include <Eigen/Core>



namespace DeFillet {

    FilletDetector::FilletDetector(SurfaceMesh* mesh, FilletDetectorParameters& parameters) {

        mesh_ = new SurfaceMesh(*mesh);

        box_ = mesh_->bounding_box();

        parameters_ = parameters;

        // Precompute and cache: dihedral angles for all edges.
        // The property key "e:dihedral-angle" stores the dihedral angle per edge.
        dihedral_angle_  =  mesh_->edge_property<float>("e:dihedral-angle");

        // Precompute and cache: centroid distance between the two adjacent triangles of each edge.
        // The property key "e:centroid_distance" stores the distance between face centroids.
        centroid_distance_ = mesh_->edge_property<float>("e:centroid_distance");

        // Precompute and cache: face normals for all faces.
        face_normals_ = mesh_->face_property<easy3d::vec3>("f:normal");


        for(auto face : mesh_->faces()) {
            face_normals_[face] = mesh_->compute_face_normal(face);
        }

        for(auto face : mesh_->faces()) {

            Sites site;

            site.pos = easy3d::geom::centroid(mesh_, face);
            site.face = face;
            site.radius = box_.diagonal_length();
            site.center = easy3d::vec3(0,0,0);
            site.rate = -1.0;
            site.flag = false;
            site.label = 0;
            sites_.emplace_back(site);
        }

        // Iterate through all edges to compute dihedral angles.
        for(auto edge : mesh_->edges()){
            dihedral_angle_[edge] = -1;     // Default/invalid marker.
            centroid_distance_[edge] = -1;  // Default/invalid marker.

            // Get the two faces adjacent to this edge.
            // For a boundary edge, one of these may be invalid.
            auto f0 = mesh_->face(edge, 0);
            auto f1 = mesh_->face(edge, 1);


            // Only compute if both adjacent faces are valid (non-boundary edge).
            if(f0.is_valid() && f1.is_valid()) {

                vec3 n0 = face_normals_[f0];
                vec3 n1 = face_normals_[f1];

                // Compute the angle between the two face normals
                // and store it as the dihedral angle for this edge.
                dihedral_angle_[edge] = angle_between(n0, n1);

                vec3 c0 = sites_[f0.idx()].pos;
                vec3 c1 = sites_[f1.idx()].pos;

                // Compute the centroid distance between the two face normals
                // and store it as centroid distance for this edge
                centroid_distance_[edge] = (c0 - c1).norm();
            }
        }

    }

    void FilletDetector::apply() {

        generate_voronoi_vertices();

        filter_voronoi_vertices();

        compute_voronoi_vertices_density_field();

        rolling_ball_trajectory_transform();

        compute_fillet_radius_field();

        compute_fillet_radius_rate_field();

        // rate_field_smoothing();

        graph_cut();
    }

    void FilletDetector::generate_voronoi_vertices() {

        std::cout << "Start generating Voronoi vertices..." <<std::endl;

        easy3d::StopWatch sw;

        voronoi_vertices_.clear();

        std::map<SurfaceMesh::Face, std::map<SurfaceMesh::Face, std::map<SurfaceMesh::Face,std::map<SurfaceMesh::Face,int>>>> unique;

        int num_patches = parameters_.num_patches;

        if(num_patches == -1) {
            std::vector<SurfaceMesh::Face> patch;
            for(auto face : mesh_->faces()) {
                patch.emplace_back(face);
            }
            std::vector<vec3> s = face_centroids(patch);
            std::vector<std::vector<int>> snv;   // sites neighboring vertices;
            std::vector<float> vvr; // vor_vertices_radius
            std::vector<std::vector<int>> vns ; // vertices neighboring sites;
            std::vector<easy3d::vec3> vv; // voronoi vertices;

            voronoi3d(s, box_, vv, vvr,snv, vns);

            for(int i = 0; i < vv.size(); i++) {
                VoronoiVertices v;
                v.pos = vv[i];
                v.radius = vvr[i];
                std::vector<easy3d::SurfaceMesh::Face> indices;
                for(int j = 0; j < 4; j++) {
                    indices.emplace_back(vns[i][j]);
                }
                v.neigh_sites = indices;
                voronoi_vertices_.emplace_back(v);
            }
        }
        else {
            std::vector<SurfaceMesh::Face> patch_centroids;

            float max_gap = farthest_point_sampling(num_patches, patch_centroids) ;


            max_gap = 0.5 * M_PI * parameters_.radius_thr * box_.diagonal_length() * 1.2;
            int idx = 0;

            omp_lock_t lock;
            omp_init_lock(&lock);

#pragma omp parallel for
            for(int i = 0; i < num_patches; i++) {

                std::vector<SurfaceMesh::Face> patch;

                crop_local_patch(patch_centroids[i], max_gap, patch);



                std::vector<vec3> ls = face_centroids(patch); // local sites;



                if(ls.size() < 10)
                    continue;



                std::vector<std::vector<int>> lsnv;   // local sites neighboring vertices;
                std::vector<float> lvvr; // vor_vertices_radius
                std::vector<std::vector<int>> lvns ; // local vertices neighboring sites;
                std::vector<easy3d::vec3> lvv; // local voronoi vertices;


                voronoi3d(ls, box_, lvv, lvvr,lsnv, lvns);


                for(size_t j = 0; j < lvv.size(); j++) {

                    std::vector<SurfaceMesh::Face> indices(4);
                    for(int k = 0; k < 4; k++) {
                        indices[k] = patch[lvns[j][k]];
                    }
                    std::sort(indices.begin(), indices.end());
                    bool flag = true;
                    if(unique.find(indices[0]) != unique.end()) {
                        auto& sub_mp1 = unique[indices[0]];
                        if(sub_mp1.find(indices[1]) != sub_mp1.end()) {
                            auto& sub_mp2 = sub_mp1[indices[1]];
                            if(sub_mp2.find(indices[2]) != sub_mp2.end()) {
                                auto& sub_mp3 = sub_mp2[indices[2]];
                                if(sub_mp3.find(indices[3]) != sub_mp3.end()) {
                                    flag = false;
                                }
                            }
                        }
                    }
                    if(flag && box_.contains(lvv[j])) {
                        omp_set_lock(&lock);

                        VoronoiVertices vv;
                        vv.pos = lvv[j];
                        vv.radius = lvvr[j];
                        vv.neigh_sites = indices;
                        voronoi_vertices_.emplace_back(vv);

                        unique[indices[0]][indices[1]][indices[2]][indices[3]] = idx++;
                        omp_unset_lock(&lock);
                    }
                }
            }
        }

        std::cout << "Successfully generated Voronoi vertices! Voronoi vertices number = " <<voronoi_vertices_.size() <<std::endl;
        std::cout << "The consumption time of Voronoi vertices generatioin: " << sw.elapsed_seconds(5) << std::endl;

    }

    void FilletDetector::filter_voronoi_vertices() {

        std::cout << "Start filtering Voronoi_vertices..." <<std::endl;

        easy3d::StopWatch sw;

        std::vector<VoronoiVertices> tmp;

        int num_radius = 0;

        //Rule_1: radius threshold
        float radius_thr = parameters_.radius_thr * box_.diagonal_length();



        for(int i = 0; i < voronoi_vertices_.size(); i++) {

            float radius = voronoi_vertices_[i].radius;
            if(radius < radius_thr) {
                tmp.emplace_back(voronoi_vertices_[i]);
            }
            else
                ++num_radius;

        }
        std::cout << "Filtered out by radius threshold: " << num_radius <<std::endl;

        std::vector<vec4> points(tmp.size());
        std::vector<bool>labels(tmp.size());

        int num_denisty = 0;

        for(int i = 0; i < tmp.size(); i++) {

            points[i] = vec4(tmp[i].pos.x,tmp[i].pos.y, tmp[i].pos.z, tmp[i].radius);
            labels[i] = true;

        }

        int num_sor_neighbors = parameters_.num_sor_neighbors;
        int num_sor_iter = parameters_.num_sor_iter;
        float num_sor_ratio = parameters_.num_sor_std_ratio;

        sor(points, labels, num_sor_neighbors,num_sor_iter, num_sor_ratio);

        voronoi_vertices_.clear();
        voronoi_vertices_.reserve(tmp.size() / 2);

        for(int i = 0; i < tmp.size(); i++) {

            if(labels[i])
                voronoi_vertices_.emplace_back(tmp[i]);
            else
                ++num_denisty;

        }
        voronoi_vertices_.shrink_to_fit();
        std::cout << "Filtered out by first density trick: " << num_denisty <<std::endl;


        tmp.clear();
        tmp.reserve(voronoi_vertices_.size() / 2);
        //Rule_2 & Rule_3: osculation condition----connected, non-tangented, genus = 1
        float eps = parameters_.epsilon;
        int num_connected = 0, num_tangented = 0, num_genus = 0;

        omp_lock_t lock;
        omp_init_lock(&lock);
#pragma omp parallel for
        for(int i = 0; i < voronoi_vertices_.size(); i++) {

            float thickness = voronoi_vertices_[i].radius * eps;
            int state_code = check_osculation_condition(voronoi_vertices_[i], thickness);
            omp_set_lock(&lock);
            if(state_code == 0) {
                tmp.emplace_back(voronoi_vertices_[i]);
            }
            else if(state_code == 1) {
                ++num_connected;
            }
            else if(state_code == 2 || state_code == 4) {
                ++num_tangented;
            }
            else if(state_code == 3) {
                ++num_genus;
            }
            omp_unset_lock(&lock);
        }
        tmp.shrink_to_fit();

        std::cout << "Filtered out by connected condition: " << num_connected <<std::endl;
        std::cout << "Filtered out by tangented condition: " << num_tangented <<std::endl;
        std::cout << "Filtered out by genus condition: " << num_genus <<std::endl;


        num_denisty = 0;
        points.resize(tmp.size());
        labels.resize(tmp.size());

        for(int i = 0; i < tmp.size(); i++) {
            points[i] = vec4(tmp[i].pos.x,tmp[i].pos.y, tmp[i].pos.z, tmp[i].radius);
            labels[i] = true;
        }

        int num = std::max((int)(num_sor_iter / 3), 1);
        sor(points, labels, num_sor_neighbors, num, num_sor_ratio);

        voronoi_vertices_.clear();
        voronoi_vertices_.reserve(tmp.size());
        for(int i = 0; i < tmp.size(); i++) {
            if(labels[i])
                voronoi_vertices_.emplace_back(tmp[i]);
            else
                ++num_denisty;
        }
        voronoi_vertices_.shrink_to_fit();
        std::cout << "Filtered out by second density trick: " << num_denisty <<std::endl;


        std::cout << "Remaining Voronoi vertices: " << voronoi_vertices_.size() <<std::endl;
        std::cout << "The consumption time of filtering Voronoi vertices: " << sw.elapsed_seconds(5) << std::endl;

    }

    void FilletDetector::compute_voronoi_vertices_density_field() {

        std::cout << "Start computing Voronoi vertices density field..." <<std::endl;

        easy3d::StopWatch sw;

        int num = voronoi_vertices_.size();

        std::vector<KNN::Point4D> knn_data;
        std::vector<easy3d::vec4> data;

        for(int i = 0; i < num; i++) {
            auto& pos = voronoi_vertices_[i].pos;
            float radius = voronoi_vertices_[i].radius;
            knn_data.emplace_back(KNN::Point4D(pos.x,pos.y, pos.z, radius));
            data.emplace_back(easy3d::vec4(pos.x,pos.y, pos.z, radius));
        }

        KNN::KdSearch4D kds(knn_data);
        int num_neighbors = parameters_.num_neighbors;
        int sigma = parameters_.sigma;

#pragma omp parallel for
        for(int i = 0; i < num; i++) {
            std::vector<size_t> indices;
            std::vector<double> dist;
            kds.kth_search(knn_data[i], num_neighbors, indices, dist);
            float density = 0;
            double w = 0;
            std::vector<vec4> points;
            for(size_t j = 0; j < indices.size(); j++) {
                int idx = indices[j];
                double dis = fabs((data[idx] - data[i]).norm());
                double val = gaussian_kernel(dis, sigma);
                density += val * dis;
                w += val;
                points.emplace_back(data[idx]);
            }
            voronoi_vertices_[i].density =  density / w;

            voronoi_vertices_[i].axis = axis_direction(points);
        }

        float max_value = 0, min_value = box_.diagonal_length();

        for(int i = 0; i < num; i++) {
            max_value = max(max_value, voronoi_vertices_[i].density);
            min_value = min(min_value, voronoi_vertices_[i].density);
        }
        std::cout << "max_value: " << max_value << std::endl;
        std::cout << "min_value: " << min_value << std::endl;

        std::cout << "The consumption time of computing Voronoi vertices density field: " << sw.elapsed_seconds(5) << std::endl;
    }

    void FilletDetector::rolling_ball_trajectory_transform() {

        std::cout << "Start rolling-ball trajectory transform..." <<std::endl;

        easy3d::StopWatch sw;
        omp_lock_t lock;
        omp_init_lock(&lock);
        float eps = parameters_.epsilon;

        std::vector<float> min_value(sites_.size(), std::numeric_limits<float>::max());
        std::vector<int> min_id(sites_.size(), -1);
        std::vector<int> count(sites_.size(), 0);
        for(int i = 0; i < voronoi_vertices_.size(); i++) {
            auto& corr_sites = voronoi_vertices_[i].corr_sites;

            for(int j = 0; j < corr_sites.size(); j++) {
                auto idx = corr_sites[j].idx();
                if(min_value[idx] > voronoi_vertices_[i].density) {
                    min_value[idx] = voronoi_vertices_[i].density;
                    min_id[idx] = i;
                }
                ++count[idx];
            }
        }


#pragma omp parallel for
        for(int i = 0; i < sites_.size(); i++) {
            if(min_id[i] != -1 && count[i] > 3) { // count[i] > 3 is trick for stable
                int id = min_id[i];
                sites_[i].center = voronoi_vertices_[id].pos;
                sites_[i].radius = voronoi_vertices_[id].radius;
                sites_[i].axis = voronoi_vertices_[id].axis;
                sites_[i].flag = true;
            }
        }


        std::cout << "The consumption time of rolling-ball trajectory transform: " << sw.elapsed_seconds(5) << std::endl;


    }

    void FilletDetector::compute_fillet_radius_field() {
        std::cout << "Start computing  fillet radius field..." <<std::endl;

        easy3d::StopWatch sw;
        int num = sites_.size();
        std::vector<float> field(num, 0);
        std::vector<float> count(num, 0);

        float angle_thr = parameters_.angle_thr;
        float eps = parameters_.epsilon;
        omp_lock_t lock;
        omp_init_lock(&lock);

#pragma omp parallel for
        for(int i = 0; i < sites_.size(); i++) {

            if(!sites_[i].flag)
                continue;

            std::set<SurfaceMesh::Face> vis;

            std::queue<SurfaceMesh::Face> que;
            easy3d::vec3 center = sites_[i].center;
            float r = sites_[i].radius;
            float thr = r * eps;

            que.push(sites_[i].face);
            while(!que.empty()) {

                SurfaceMesh::Face cur = que.front(); que.pop();

                for(auto halfedge : mesh_->halfedges(cur)) {
                    auto opp_face = mesh_->face(mesh_->opposite(halfedge));
                    auto edge = mesh_->edge(halfedge);

                    if(opp_face.is_valid() && vis.find(opp_face) == vis.end() && dihedral_angle_[edge] < angle_thr) {
                        float err =  fabs((sites_[opp_face.idx()].pos - center).norm() -  r);
                        if(err < thr) {
                            que.push(opp_face);
                            vis.insert(opp_face);
                        }
                    }
                }
            }

            for(auto face : vis) {
                omp_set_lock(&lock);
                field[face.idx()] += (sites_[face.idx()].pos - center).norm();
                ++count[face.idx()];
                omp_unset_lock(&lock);
            }
        }

        for(int i = 0; i < num; i++) {
            if(count[i] > 0) {
                field[i] /= count[i];
                sites_[i].radius = field[i];
            }
            else {
                sites_[i].radius = 0;
            }
        }

        std::cout << "The consumption time of computing  fillet radius field: " << sw.elapsed_seconds(5) << std::endl;
    }

    void FilletDetector::compute_fillet_radius_rate_field() {
        std::cout << "Start computing fillet radius rate field..." <<std::endl;

        easy3d::StopWatch sw;
        float angle_thr = parameters_.angle_thr;
        for(auto face : mesh_->faces()) {
            if(sites_[face.idx()].radius < 1e-6) {
                sites_[face.idx()].rate = 1.0;
                continue;
            }

            float rate = 0;
            int count = 0;
            for(auto halfedge : mesh_->halfedges(face)) {

                auto opp_face = mesh_->face(mesh_->opposite(halfedge));
                auto edge = mesh_->edge(halfedge);

                if(opp_face.is_valid() && dihedral_angle_[edge] < angle_thr) {

                    float dis = centroid_distance_[edge];
                    float radius_diff = fabs(sites_[face.idx()].radius - sites_[opp_face.idx()].radius);
                    rate += radius_diff / dis;
                    ++count;
                }
            }

            if(count) {
                sites_[face.idx()].rate = min(rate / count, 1.0f);
                // sites_[face.idx()].rate = rate / count;
            }
            else {
                sites_[face.idx()].rate = 1.0;
            }
        }

        std::cout << "The consumption time of computing fillet radius rate field: " << sw.elapsed_seconds(5) << std::endl;
    }

    void FilletDetector::rate_field_smoothing() {
        std::cout << "Start smoothing rate field..." <<std::endl;
        easy3d::StopWatch sw;
        float angle_thr = parameters_.angle_thr;
        int num_iter = parameters_.num_smooth_iter;
        for(int iter = 0; iter < num_iter; iter++) {
            std::vector<Sites> tmp = sites_;
            for(int i = 0; i < sites_.size(); i++) {

                auto face = sites_[i].face;
                float val = 0;
                int count = 0;
                for(auto halfedge : mesh_->halfedges(face)) {

                    auto opp_face = mesh_->face(mesh_->opposite(halfedge));
                    auto edge = mesh_->edge(halfedge);

                    if(opp_face.is_valid() && dihedral_angle_[edge] < angle_thr) {
                        val += sites_[opp_face.idx()].rate;
                        ++count;
                    }
                }
                if(count)
                    tmp[i].rate = val / count;

            }
            sites_ = tmp;
        }

        std::cout << "The consumption time of smoothing rate field: " << sw.elapsed_seconds(5) << std::endl;

    }

    void FilletDetector::graph_cut() {

        std::cout << "Start running graph cut..." <<std::endl;
        easy3d::StopWatch sw;

        int num_node = sites_.size();
        double lamdba = parameters_.lamdba;
        float angle_thr = parameters_.angle_thr;
        std::vector<double> data_cost(2 * num_node);
        std::vector<std::pair<int,int>> edges;
        std::vector<double> edge_weights;

        for(int i = 0; i < num_node; i++) {
            double rate = sites_[i].rate;
            data_cost[i] = (1.0 - rate) / num_node / 2;
            data_cost[i + num_node] = rate / num_node / 2;
        }

        float edge_sum = 0;
        for(auto edge : mesh_->edges()) {
            auto face0 = mesh_->face(edge, 0);
            auto face1 = mesh_->face(edge, 1);
            if(face0.is_valid() && face1.is_valid()) {
                float edge_len = mesh_->edge_length(edge);
                int id0 = face0.idx(), id1 = face1.idx();
                edges.emplace_back(std::make_pair(id0, id1));
                edge_sum += edge_len;
            }
        }


        for(auto edge : mesh_->edges()) {
            auto face0 = mesh_->face(edge, 0);
            auto face1 = mesh_->face(edge, 1);
            if(face0.is_valid() && face1.is_valid() && dihedral_angle_[edge] < angle_thr) {
                float edge_len = mesh_->edge_length(edge);
                double w = edge_len * lamdba / edge_sum;
                edge_weights.emplace_back(w);
            }
            else {
                edge_weights.emplace_back(0);
            }
        }

        GCoptimizationGeneralGraph gc(num_node, 2);
        for(int i = 0; i < edges.size(); i++) {
            gc.setNeighbors(edges[i].first, edges[i].second);
            // gc.setNeighbors(edges[i].second, edges[i].first);
        }

        GCP::DataCost data_item(data_cost, num_node, 2);
        GCP::SmoothCost smooth_item(edges, edge_weights);
        gc.setDataCostFunctor(&data_item);
        gc.setSmoothCostFunctor(&smooth_item);
        std::cout << "Before optimization energy is " << gc.compute_energy() << std::endl;
        gc.expansion(10000);
        std::cout << "After optimization energy is " << gc.compute_energy() << std::endl;

#pragma omp parallel for
        for(int i = 0; i < num_node; i++) {
            int label = gc.whatLabel(i);
            sites_[i].label = label;
        }

        std::cout << "The consumption time of running graph cut: " << sw.elapsed_seconds(5) << std::endl;

    }

    PointCloud* FilletDetector::voronoi_vertices() const {

        PointCloud* cloud = new PointCloud;

        int num = voronoi_vertices_.size();

        for(int i = 0; i < num; i++) {
            cloud->add_vertex(voronoi_vertices_[i].pos);
        }

        auto nor = cloud->add_vertex_property<vec3>("v:normal");

        for(auto  v : cloud->vertices()) {
            nor[v] = voronoi_vertices_[v.idx()].axis;
        }

        return cloud;
    }

    PointCloud* FilletDetector::rolling_ball_centers() const {
        PointCloud* cloud = new PointCloud;
        auto nor = cloud->add_vertex_property<vec3>("v:normal");
        int num = sites_.size();

        for(int i = 0; i < num; i++) {
            if(sites_[i].flag) {
                auto v = cloud->add_vertex(sites_[i].center);
                nor[v] = sites_[i].axis;
            }
        }

        return cloud;
    }

    std::vector<float> FilletDetector::radius_rate_field() const {

        std::vector<float> field;
        for(int i = 0; i < sites_.size(); i++) {
            field.emplace_back(sites_[i].rate);
        }

        return field;
    }


    std::vector<float> FilletDetector::radius_field() const {
        std::vector<float> field;
        for(int i = 0; i < sites_.size(); i++) {
            // if(sites_[i].flag)
                field.emplace_back(sites_[i].radius);
        }

        return field;
    }

    std::vector<int> FilletDetector::fillet_labels() const {
        std::vector<int> labels;

        for(int i = 0; i < sites_.size(); i++) {
            labels.emplace_back(sites_[i].label);
        }

        return labels;
    }


    float FilletDetector::farthest_point_sampling(int num_samples,
                                                  std::vector<SurfaceMesh::Face>& indices) {

        int num = mesh_->n_faces();

        indices.clear();
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> distribution(0, num - 1);

        std::vector<float> dis(num, std::numeric_limits<float>::max());

        for(int i = 0; i < num_samples; i++) {

            std::priority_queue<std::pair<float, SurfaceMesh::Face>> que;
            SurfaceMesh::Face sface;
            if(i == 0) {
                sface = SurfaceMesh::Face(distribution(gen));
            } else {
                float maxx = 0.0;
                int idx = 0;
                for(int j = 0; j < num; j++) {
                    SurfaceMesh::Face f(j);
                    if(dis[j] > maxx) {
                        maxx = dis[j]; idx = j;
                    }
                }
                sface = SurfaceMesh::Face(idx);
            }

            indices.emplace_back(sface);

            dis[sface.idx()] = 0.0;
            que.push(std::make_pair(0.0, sface));

            std::set<SurfaceMesh::Face> vis;
            float angle_thr = parameters_.angle_thr;
            while(!que.empty()) {

                auto cur_dis = que.top().first;
                auto cur_face = que.top().second;
                que.pop();

                int cur_fid = cur_face.idx();

                if(dis[cur_fid] < -cur_dis) {
                    continue;
                }

                dis[cur_fid] = -cur_dis;


                for(auto halfedge : mesh_->halfedges(cur_face)) {
                    auto opp_face = mesh_->face(mesh_->opposite(halfedge));
                    auto edge = mesh_->edge(halfedge);

                    if(opp_face.is_valid() && vis.find(opp_face) == vis.end()) {
                        float val = centroid_distance_[edge];
                        if(dis[opp_face.idx()] > -cur_dis + val && dihedral_angle_[edge] < angle_thr) {
                            que.push(std::make_pair(cur_dis - val, opp_face));
                            vis.insert(opp_face);
                        }
                    }
                }
            }
        }

        float max_gap = 0.0;
        for(int i = 0; i < num; i++) {
            SurfaceMesh::Face f(i);
            if(dis[i] > max_gap) {
                max_gap = dis[i];
            }
        }

        return max_gap;

    }

    void FilletDetector::crop_local_patch(SurfaceMesh::Face cface,
                                          float max_gap, std::vector<SurfaceMesh::Face>& patch) {

        std::set<SurfaceMesh::Face> vis;
        std::priority_queue<std::pair<float, SurfaceMesh::Face>> que;
        que.push(std::make_pair(0.0, cface));

        float angle_thr = parameters_.angle_thr;

        while(!que.empty()) {

            auto cur_dis = que.top().first;
            auto cur_face = que.top().second;
            que.pop();

            for(auto halfedge : mesh_->halfedges(cur_face)) {
                auto opp_face = mesh_->face(mesh_->opposite(halfedge));

                if(vis.find(opp_face) == vis.end()) {
                    auto edge = mesh_->edge(halfedge);
                    float val = centroid_distance_[edge];
                    if(max_gap > -cur_dis + val && dihedral_angle_[edge] < angle_thr) {
                        que.push(std::make_pair(cur_dis - val, opp_face));
                        vis.insert(opp_face);
                    }
                }
            }

        }

        patch.insert(patch.end(), vis.begin(), vis.end());
    }

    int FilletDetector::check_osculation_condition(VoronoiVertices& vor, float thickness) const {

        std::vector<SurfaceMesh::Face>& neigh_sites = vor.neigh_sites;
        std::vector<SurfaceMesh::Face>& corr_sites = vor.corr_sites;
        vec3 pos = vor.pos;
        float radius = vor.radius;
        float thr = thickness * 0.5;

        std::set<SurfaceMesh::Face> vis;
        std::queue<SurfaceMesh::Face> que;

        que.push(neigh_sites[0]);
        float angle_thr = parameters_.angle_thr;
        while(!que.empty()) {

            SurfaceMesh::Face cur_face = que.front(); que.pop();

            if(vis.find(cur_face) != vis.end())
                continue;

            vis.insert(cur_face);

            for(auto halfedge : mesh_->halfedges(cur_face)) {
                auto opp_face = mesh_->face(mesh_->opposite(halfedge));
                auto edge = mesh_->edge(halfedge);
                if(opp_face.is_valid() && vis.find(opp_face) == vis.end() && dihedral_angle_[edge] < angle_thr) {
                    vec3 opp_pos = sites_[opp_face.idx()].pos;
                    float err = fabs((pos - opp_pos).norm() - radius);
                    if(err < thr) {
                        que.push(opp_face);
                    }
                }
            }
        }

        // Check if corr_stites are connected
        for(size_t i = 0; i < neigh_sites.size(); i++) {
            if(vis.find(neigh_sites[i]) == vis.end()) {
                return 1;
            }
        }

        // Check if corr_stites are tangented to surface
        std::vector<SurfaceMesh::Edge> boundaries;

        for(size_t i = 0; i < neigh_sites.size(); i++) {
            for(auto halfedge : mesh_->halfedges(neigh_sites[i])) {
                SurfaceMesh::Edge edge  = mesh_->edge(halfedge);
                boundaries.emplace_back(edge);
            }
        }

        if(count_boundaries_components(boundaries) < 2 ) {
            return 2;
        }


        boundaries.clear();
        //Check if corr_stites's genus = 1
        std::vector<SurfaceMesh::Face> tmp(vis.begin(), vis.end());
        for(int i = 0; i < tmp.size(); i++) {
            SurfaceMesh::Face cur = tmp[i];
            for(auto halfedge : mesh_->halfedges(cur)) {

                auto opp_face = mesh_->face(mesh_->opposite(halfedge));

                if(opp_face.is_valid() && vis.find(opp_face) == vis.end()) {

                    SurfaceMesh::Edge edge  = mesh_->edge(halfedge);
                    boundaries.emplace_back(edge);

                }
            }
        }

        if(count_boundaries_components(boundaries) > 1) {
            return 3;
        }


        corr_sites = tmp;

        return 0;

    }


    int FilletDetector::count_boundaries_components(std::vector<SurfaceMesh::Edge>& boundaries) const {

        std::map<SurfaceMesh::Vertex, int> mp;
        int id = 0;

        for(size_t i = 0; i < boundaries.size(); i++) {
            SurfaceMesh::Edge edge  = boundaries[i];

            SurfaceMesh::Vertex v0 = mesh_->vertex(edge, 0);
            SurfaceMesh::Vertex v1 = mesh_->vertex(edge, 1);

            if(mp.find(v0) == mp.end()) {
                mp[v0] = id++;
            }

            if(mp.find(v1) == mp.end()) {
                mp[v1] = id++;
            }
        }

        std::vector<int> bcj(id);

        function<int(int)> find = [&](int x){ return bcj[x]==x ? x : bcj[x]=find(bcj[x]); };

        std::iota(bcj.begin(), bcj.end(), 0);

        for(size_t i = 0; i < boundaries.size(); i++) {
            SurfaceMesh::Edge edge  = boundaries[i];

            SurfaceMesh::Vertex v0 = mesh_->vertex(edge, 0);
            SurfaceMesh::Vertex v1 = mesh_->vertex(edge, 1);

            int id0 = mp[v0], id1 = mp[v1];

            bcj[find(id0)] = find(id1);
        }

        int num_components = 0;
        for(size_t i = 0; i < bcj.size(); i++) {
            if(bcj[i] == i)
                 num_components++;
        }

        return num_components;
    }


    std::vector<vec3> FilletDetector::face_centroids(std::vector<SurfaceMesh::Face>& faces) const{

        std::vector<vec3> centroids;

        for(size_t i = 0; i < faces.size(); i++) {
            vec3 p = sites_[faces[i].idx()].pos;
            centroids.emplace_back(p);
        }

        return centroids;
    }


}
