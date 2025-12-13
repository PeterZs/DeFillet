//
// Created by xiaowuga on 2025/12/9.
//

#include <utils.h>

#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/util/file_system.h>

#include <omp.h>

#include <CLI/CLI.hpp>


int main(int argc, char **argv) {
    std::string config_path;
    CLI::App app{"The Removal of DeFillet Command Line"};
    app.add_option("-c,--config", config_path, "Configure file")->required();
    CLI11_PARSE(app, argc, argv);

    try {
        auto params = DeFillet::load_remover_config(config_path);

        std::cout << "Input path: " << params.input_path << std::endl;
        std::cout << "Output dir: " << params.out_dir << std::endl;
        std::cout << "angle_thr: " << params.angle_thr << std::endl;
        std::cout << "beta_e: " << params.beta_e << std::endl;
        std::cout << "beta_f: " << params.beta_f << std::endl;
        std::cout << "beta_c: " << params.beta_c << std::endl;
        std::cout << "num_opt_iter: " << params.num_opt_iter << std::endl;

        int num_threads = omp_get_max_threads();
        if(params.num_threads == -1)
            params.num_threads = num_threads;

        params.num_threads = min(num_threads, params.num_threads);
        std::cout << "num_threads: " << params.num_threads << std::endl;

        omp_set_num_threads(params.num_threads);

        easy3d::SurfaceMesh* mesh = easy3d::SurfaceMeshIO::load(params.input_path);
        std::string label_path = params.label_path;
        std::vector<int> labels = DeFillet::load_fillet_labels(label_path);
        DeFillet::FilletRemover remover;
        std::cout << labels.size() << ' ' << mesh->n_faces() << std::endl;
        remover.initialize(mesh, params, labels);


        remover.optimize();
        if(!easy3d::file_system::is_directory(params.out_dir)) {
            easy3d::file_system::create_directory(params.out_dir);
        }

        std::string base_name = easy3d::file_system::base_name(params.input_path);
        std::string res_dir = params.out_dir + "/" + base_name + "_removal_" + DeFillet::get_time_stamp(true);

        easy3d::file_system::create_directory(res_dir);


        DeFillet::save_remover_config(params, res_dir + "/" + base_name + "_param.json");

        easy3d::SurfaceMesh* defillet_mesh = remover.defillet_mesh_;
        std::string defillet_mesh_path = res_dir + "/" + base_name + "_defillet.ply";
        easy3d::SurfaceMeshIO::save(defillet_mesh_path, defillet_mesh);

        easy3d::SurfaceMesh* focus_mesh = remover.focus_area_;
        std::string focus_mesh_path = res_dir + "/" + base_name + "_focus.ply";
        easy3d::SurfaceMeshIO::save(focus_mesh_path, focus_mesh);

        easy3d::SurfaceMesh* fillet_mesh = remover.fillet_mesh_;
        std::string fillet_mesh_path = res_dir + "/" + base_name + "_fillet.ply";
        easy3d::SurfaceMeshIO::save(fillet_mesh_path, fillet_mesh);

        easy3d::SurfaceMesh* non_fillet_mesh = remover.non_fillet_mesh_;
        std::string non_fillet_mesh_path = res_dir + "/" + base_name + "_non_fillet.ply";
        easy3d::SurfaceMeshIO::save(non_fillet_mesh_path, non_fillet_mesh);

        auto tar_normals = remover.tar_normals;
        std::string tar_normal_path = res_dir + "/" + base_name + "_tar_normals.obj";
        DeFillet::save_target_normals(tar_normals, tar_normal_path);


    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }


    return 0;
}