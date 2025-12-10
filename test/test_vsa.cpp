//
// Created by xiaowuga on 2025/9/27.
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_approximation/approximate_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <boost/graph/properties.hpp>
#include <boost/graph/graph_traits.hpp>
#include <CGAL/IO/PLY.h>

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <easy3d/core/random.h>

#include <easy3d/core/surface_mesh.h>
#include <easy3d/fileio/surface_mesh_io.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel   Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3>                   Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor            face_descriptor;
typedef Mesh::Property_map<face_descriptor, std::size_t>      Face_proxy_pmap;

namespace VSA = CGAL::Surface_mesh_approximation;



int main() {
    const std::string filename = "D:\\code\\DeFillet\\data\\091_wheel_assembly_part.ply";

    Mesh mesh;
    if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh)) {
        std::cerr << "Invalid input file." << std::endl;
        return EXIT_FAILURE;
      }

    std::cout << "Number of vertices: " << num_vertices(mesh) << std::endl;
    std::cout << "Number of faces: " << num_faces(mesh) << std::endl;

    std::vector<Kernel::Point_3> anchors;
    std::vector<std::array<std::size_t, 3> > triangles;

    Face_proxy_pmap fpxmap =
      mesh.add_property_map<face_descriptor, std::size_t>("f:proxy_id", 0).first;

    // output planar proxies
    std::vector<Kernel::Vector_3> proxies;

    // free function interface with named parameters
    VSA::approximate_triangle_mesh(mesh,
      CGAL::parameters::min_error_drop(0.01). // seeding with minimum error drop
                        number_of_iterations(100). // set number of clustering iterations after seeding
                        subdivision_ratio(0.5). // set chord subdivision ratio threshold when meshing
                        face_proxy_map(fpxmap). // get face partition map
                        proxies(std::back_inserter(proxies)). // output proxies
                        anchors(std::back_inserter(anchors)). // output anchor points
                        triangles(std::back_inserter(triangles))); // output indexed triangles

  easy3d::SurfaceMesh* emesh = new  easy3d::SurfaceMesh;
  for(auto v : mesh.vertices()) {
    auto vdata = mesh.point(v);
    easy3d::vec3 ev(vdata.x(), vdata.y(),vdata.z());
    emesh->add_vertex(ev);
  }

  for (auto face : mesh.faces()) {
    auto halfedge = mesh.halfedge(face);
    std::vector<easy3d::SurfaceMesh::Vertex> tmp;
    auto it = halfedge;
    do {
      int id = mesh.source(it).id();
      tmp.emplace_back(easy3d::SurfaceMesh::Vertex(id));
      it = mesh.next(it);
    } while(it != halfedge);
    emesh->add_face(tmp);
  }

  // easy3d::SurfaceMeshIO::save("../out/asd.ply", emesh);
  std::vector<easy3d::vec3> corlors(proxies.size());

  for(int i = 0; i < corlors.size(); i++) {
    corlors[i] = easy3d::random_color();
  }

  auto color_map = emesh->add_face_property<easy3d::vec3>("f:color");
  for (auto face : mesh.faces()) {
    std::size_t proxy_id = fpxmap[face];  // Get the proxy ID for the face
    easy3d::vec3 color = corlors[proxy_id];
    // Assign this color to the face
    easy3d::SurfaceMesh::Face f(face.id());
    color_map[f] = color;
  }

  easy3d::SurfaceMeshIO::save("../out/asd.ply", emesh);
  return EXIT_SUCCESS;
}
