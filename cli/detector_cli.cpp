//
// Created by xiaowuga on 2025/8/28.
//
#include <fillet_detector.h>
#include <utils.h>

#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/util/file_system.h>

#include <omp.h>

#include <CLI/CLI.hpp>


int main(int argc, char **argv) {
    std::string config_path;
    CLI::App app{"The Detector of DeFillet Command Line"};
    app.add_option("-c,--config", config_path, "Configure file")->required();
    CLI11_PARSE(app, argc, argv);

    try {
        auto params = DeFillet::load_detector_config(config_path);

        std::cout << "Input path: " << params.input_path << std::endl;
        std::cout << "Output dir: " << params.out_dir << std::endl;
        std::cout << "epsilon: " << params.epsilon << std::endl;
        std::cout << "radius_thr: " << params.radius_thr << std::endl;
        std::cout << "lambda: " << params.lamdba << std::endl;
        std::cout << "angle_thr: " << params.angle_thr << std::endl;
        std::cout << "sigma: " << params.sigma << std::endl;
        std::cout << "num_patches: " << params.num_patches << std::endl;
        std::cout << "num_neighbors: " << params.num_neighbors << std::endl;
        std::cout << "num_smooth_iter: " << params.num_smooth_iter << std::endl;
        std::cout << "num_sor_iter: " << params.num_sor_iter << std::endl;
        std::cout << "num_sor_neighbors: " << params.num_sor_neighbors << std::endl;
        std::cout << "num_sor_std_ratio: " << params.num_sor_std_ratio << std::endl;

        int num_threads = omp_get_max_threads();
        if(params.num_threads == -1)
            params.num_threads = num_threads;

        params.num_threads = min(num_threads, params.num_threads);
        std::cout << "num_threads: " << params.num_threads << std::endl;

        omp_set_num_threads(params.num_threads);

        easy3d::SurfaceMesh* mesh = easy3d::SurfaceMeshIO::load(params.input_path);

        easy3d::vec3 centroids;
        double scale;
        DeFillet::normalize_model(mesh,  centroids, scale);

        DeFillet::FilletDetector detector(mesh, params);


        detector.apply();

        if(!easy3d::file_system::is_directory(params.out_dir)) {
            easy3d::file_system::create_directory(params.out_dir);
        }

        // DeFillet::inverse_normalize_model(mesh, centroids, scale);
        std::string base_name = easy3d::file_system::base_name(params.input_path);
        std::string res_dir = params.out_dir + "/" + base_name + "_detector_" + DeFillet::get_time_stamp(true);

        easy3d::file_system::create_directory(res_dir);
        easy3d::PointCloud* vv = detector.voronoi_vertices();
        easy3d::PointCloudIO::save(res_dir + "/"+ base_name +"_voronoi_vertices.ply" , vv);

        easy3d::PointCloud* center = detector.rolling_ball_centers();
        easy3d::PointCloudIO::save(res_dir + "/"+ base_name +"_rolling_ball_centers.ply" , center);

        std::vector<float> radius_field = detector.radius_field();
        DeFillet::save_field(mesh, radius_field, res_dir + "/"+ base_name +"_radius_field.ply");

        std::vector<float> rate_field = detector.radius_rate_field();

        DeFillet::save_field(mesh, rate_field, res_dir + "/"+ base_name +"_rate_field.ply");

        std::vector<int> fillet_labels = detector.fillet_labels();

        DeFillet::save_fillet_segmentation(mesh, fillet_labels, res_dir + "/"+ base_name +"_seg.ply");

        DeFillet::save_detector_config(params, res_dir + "/" + base_name + "_param.json");

        DeFillet::save_fillet_labels(fillet_labels, res_dir + "/" + base_name + "_seg.json");



    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }


    return 0;
}