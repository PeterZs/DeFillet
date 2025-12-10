//
// Created by 13900K on 2024/3/18.
//

#ifndef DEFILLET_SURAFCE_MESH_SEGMENTER_H
#define DEFILLET_SURAFCE_MESH_SEGMENTER_H

#include <easy3d/core/surface_mesh.h>

namespace easy3d {
    class SurfaceMeshSegmenter {
    public:
        SurfaceMeshSegmenter(easy3d::SurfaceMesh* mesh);
        ~SurfaceMeshSegmenter();

        template<typename FT>
        SurfaceMesh* segment(const SurfaceMesh::FaceProperty<FT>& segments, FT label) {
            std::set<easy3d::SurfaceMesh::Face> faces_set;
            std::set<easy3d::SurfaceMesh::Vertex> points_set;
            for(auto f : mesh_->faces()) {
                if(segments[f] == label) {
                    faces_set.insert(f);
                    for(auto v : mesh_->vertices(f)) {
                        points_set.insert(v);
                    }
                }
            }

            std::map<int, size_t> mp;
            int nb_points = points_set.size();
            std::vector<int> point_map(nb_points);
            size_t id = 0;
            SurfaceMesh* segmented_mesh = new SurfaceMesh;
            for(auto v : points_set) {
                point_map[id] = v.idx();
                mp[v.idx()] = id++;
                easy3d::vec3 p = mesh_->position(v);
                segmented_mesh->add_vertex(p);
            }

            int nb_faces = faces_set.size();
            std::vector<int> face_map(nb_faces);
            id = 0;
            for(auto f : faces_set) {
                face_map[id] = f.idx();
                id++;
                std::vector<easy3d::SurfaceMesh::Vertex> tmp;
                for(auto v : mesh_->vertices(f)) {
                    tmp.emplace_back(easy3d::SurfaceMesh::Vertex(mp[v.idx()]));
                }
                segmented_mesh->add_triangle(tmp[0], tmp[1], tmp[2]);
            }
            auto original_point_index = segmented_mesh->vertex_property<int>("v:original_index");
            auto original_face_index = segmented_mesh->face_property<int>("f:original_index");

            for(auto v : segmented_mesh->vertices()) {
                original_point_index[v] = point_map[v.idx()];
            }

            for(auto f : segmented_mesh->faces()) {
                original_face_index[f] = face_map[f.idx()];
            }

            return segmented_mesh;
        }
    private:
        easy3d::SurfaceMesh* mesh_;
        // easy3d::SurfaceMesh* mesh_;
    };
}



#endif //DEFILLET_SURAFCE_MESH_SEGMENTER_H
