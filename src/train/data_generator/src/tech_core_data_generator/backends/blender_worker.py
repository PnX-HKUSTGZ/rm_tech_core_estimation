"""@brief Cycles worker, executed by Blender's bundled Python in a private directory."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Load only the OBJ reader. Importing the package would require the host Python dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import read_obj


def _build(config, work):
    """@brief Build the static mesh, studio and camera. All visual OBJ positions are core-frame metres at zero joints."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.cycles.max_bounces = 3
    camera = config["camera"]
    scene.render.resolution_x = camera["width"]
    scene.render.resolution_y = camera["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "None"
    scene.world = bpy.data.worlds.new("studio_world")
    scene.world.use_nodes = True
    directory = Path(config["model_dir"])
    root = ET.parse(directory / "tech_core.xml").getroot()
    meshes = {m.get("name"): directory / m.get("file") for m in root.findall("asset/mesh")}
    colors = {m.get("name"): tuple(map(float, m.get("rgba").split())) for m in root.findall("asset/material")}
    materials = {}
    for geom in root.findall(".//worldbody//geom[@group='0'][@type='mesh']"):
        name = geom.get("name")
        vertices, faces = read_obj(meshes[geom.get("mesh")])
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        material = bpy.data.materials.new(geom.get("material"))
        material.use_nodes = True
        mesh.materials.append(material)
        materials[name] = (material.node_tree.nodes["Principled BSDF"].inputs, colors[geom.get("material")])
    light = bpy.data.lights.new("key_light", "AREA")
    light.shape = "DISK"
    light.size = 3
    scene.collection.objects.link(bpy.data.objects.new("key_light", light))
    lens = bpy.data.cameras.new("dataset_camera")
    lens.sensor_fit = "VERTICAL"
    lens.sensor_height = 24
    lens.lens = lens.sensor_height / (2 * np.tan(camera["fovy"] / 2))
    lens.clip_start, lens.clip_end = camera["near"], camera["far"]
    scene.camera = bpy.data.objects.new("dataset_camera", lens)
    scene.collection.objects.link(scene.camera)
    scene.view_layers[0].use_pass_z = True
    if bpy.app.version >= (5, 0, 0):
        tree = bpy.data.node_groups.new("Dataset outputs", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        composite = tree.nodes.new("NodeGroupOutput")
        depth = tree.nodes.new("CompositorNodeOutputFile")
        depth.directory = str(work)
        depth.format.media_type = "IMAGE"
        depth.file_name = ""
        depth.file_output_items.clear()
        depth.file_output_items.new("RGBA", "depth0001")  # Match the frame suffix added by Blender 4.
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        composite = tree.nodes.new("CompositorNodeComposite")
        depth = tree.nodes.new("CompositorNodeOutputFile")
        depth.base_path = str(work)
        depth.file_slots[0].path = "depth"
    layers = tree.nodes.new("CompositorNodeRLayers")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_mode = "RGB"
    depth.format.color_depth = "32"
    tree.links.new(layers.outputs["Depth"], depth.inputs[0])
    scene.frame_set(1)
    scene.render.filepath = str(work / "rgb.png")
    return scene, materials


def _render(scene, materials, request, camera, work):
    """@brief Apply a complete frame request and transfer RGB8 and float32 optical depth (m)."""
    settings = request["appearance"]
    scene.view_settings.view_transform = settings["view_transform"]
    scene.view_settings.exposure = settings["exposure"]
    scene.render.film_transparent = settings["transparent_background"]
    background = scene.world.node_tree.nodes["Background"].inputs
    background["Color"].default_value = (*settings["world_color"], 1)
    background["Strength"].default_value = settings["world_strength"]
    light = bpy.data.objects["key_light"]
    light.data.energy = settings["light_energy"]
    light.location = settings["light_position"]
    light.rotation_euler = (Vector((-0.1, 0, 0.2)) - light.location).to_track_quat("-Z", "Y").to_euler()
    for name, (shader, rgba) in materials.items():
        shader["Metallic"].default_value = settings["metallic"]
        shader["Roughness"].default_value = settings["roughness"]
        if name in request["lamps"]:
            lamp = request["lamps"][name]
            color = (*lamp["rgb"], 1)
            shader["Emission Color"].default_value = color
            shader["Emission Strength"].default_value = 2 * settings["emission_scale"] if lamp["on"] else 0
        else:
            color = tuple(min(1, value * settings["material_scale"]) for value in rgba[:3]) + (rgba[3],)
        shader["Base Color"].default_value = color
    core_from_render_camera = np.linalg.inv(request["camera_from_core"]) @ np.diag([1, -1, -1, 1])
    scene.camera.matrix_world = Matrix(core_from_render_camera.tolist())
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(work / "depth0001.exr"), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    depth = pixels.reshape(camera["height"], camera["width"], 4)[::-1, :, 0].copy()
    depth[~np.isfinite(depth) | (depth >= camera["far"]) | (depth < camera["near"])] = 0
    np.save(work / "depth.npy", depth, allow_pickle=False)


def _main():
    """@brief Read render requests until the parent closes the pipe."""
    path = Path(sys.argv[sys.argv.index("--") + 1])
    config = json.loads(path.read_text())
    scene, materials = _build(config, path.parent)
    print("TECH_CORE_RESULT " + json.dumps({"version": bpy.app.version_string}), flush=True)
    for line in sys.stdin:
        _render(scene, materials, json.loads(line), config["camera"], path.parent)
        print("TECH_CORE_RESULT {}", flush=True)


if __name__ == "__main__":
    _main()
