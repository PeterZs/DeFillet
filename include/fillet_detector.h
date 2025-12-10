//
// Created by xiaowuga on 2025/8/16.
//

#ifndef DEFILLET_FILET_DETECTOR_H
#define DEFILLET_FILET_DETECTOR_H

#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/point_cloud.h>

using namespace easy3d;
using namespace std;


namespace DeFillet {

    class FilletDetectorParameters {

    public:
        std::string input_path;
        std::string out_dir;
        float epsilon, radius_thr, angle_thr, sigma, lamdba;

        int num_patches, num_neighbors, num_smooth_iter;
        // sor method
        int num_sor_neighbors, num_sor_iter;
        float num_sor_std_ratio;
        int num_threads;
    };

    class VoronoiVertices {
    public:
        vec3 pos;

        float radius, density;

        std::vector<SurfaceMesh::Face> neigh_sites; // neighboring sites

        std::vector<SurfaceMesh::Face> corr_sites; //corresponding sites

        vec3 axis;

        bool flag;
    };

    class Sites {
    public:
        vec3 pos;

        SurfaceMesh::Face face;

        float radius;

        float rate;


        vec3 center; //rolling-ball center

        vec3 axis;

        bool flag;

        int label;
    };



    class FilletDetector {

    public:

        FilletDetector(SurfaceMesh *mesh, FilletDetectorParameters& parameters);

        void apply();

        void generate_voronoi_vertices();

        void compute_voronoi_vertices_density_field();

        void filter_voronoi_vertices();

        void rolling_ball_trajectory_transform();

        void compute_fillet_radius_field();

        void compute_fillet_radius_rate_field();

        void rate_field_smoothing();

        void graph_cut();

    public: // setor or getor functions

        PointCloud* voronoi_vertices() const;

        PointCloud* rolling_ball_centers() const;

        std::vector<float> radius_rate_field() const;

        std::vector<float> radius_field() const;

        std::vector<int> fillet_labels() const;


    private: //utils functions

        float farthest_point_sampling(int num_samples,  std::vector<SurfaceMesh::Face>& indices);

        void crop_local_patch(SurfaceMesh::Face cface, float max_gap, std::vector<SurfaceMesh::Face>& patch);

        int check_osculation_condition(VoronoiVertices& vor, float thickness) const;

        int count_boundaries_components(std::vector<SurfaceMesh::Edge>& boundaries) const;

        std::vector<vec3> face_centroids(std::vector<SurfaceMesh::Face>& faces) const;


    public:

        SurfaceMesh* mesh_;

        Box3 box_;

        FilletDetectorParameters parameters_;

    private:

        SurfaceMesh::FaceProperty<vec3> face_normals_;

        SurfaceMesh::EdgeProperty<float> dihedral_angle_;

        SurfaceMesh::EdgeProperty<float> centroid_distance_;

        std::vector<VoronoiVertices> voronoi_vertices_;

        std::vector<Sites> sites_;


    };
}


#endif //DEFILLET_FILET_DETECTOR_H
