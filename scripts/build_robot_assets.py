"""Build a six-DOF UR3 CB3 + fixed-open Robotiq 2F-85 classroom model.

Build dependencies: NumPy, SciPy, xacro. No student-side dependencies.
UR3 source assets are already converted from installed official ur_description 2.9.0.
Robotiq source files are pinned to commit 7d0bc4aeaf16c8934520667bac4617e192a866ce.
The last wrist's official unlimited rotation is represented by a documented finite
search interval only for wrist_3_joint; this is not a hardware joint limit.
"""
from pathlib import Path
import copy
import argparse
import hashlib
import json
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import xacro

ROOT = Path(__file__).resolve().parents[1]
UR = None  # Supplied explicitly with --ur-source; not needed to serve the website.
MODELS = ROOT / 'models'
GRIP = MODELS / 'robotiq_source'
OUT = ROOT / 'dist/assets/robot'
ARM = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
COMMIT = '7d0bc4aeaf16c8934520667bac4617e192a866ce'
NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}


def vec(text):
    return np.fromstring(text, sep=' ')


def origin(e):
    return {'xyz': vec(e.get('xyz', '0 0 0') if e is not None else '0 0 0').tolist(),
            'rpy': vec(e.get('rpy', '0 0 0') if e is not None else '0 0 0').tolist()}


def transform(joint):
    o = origin(joint.find('origin'))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', o['rpy']).as_matrix()
    T[:3, 3] = o['xyz']
    return T


def collada_parts(path):
    root = ET.parse(path).getroot()
    unit = float(root.find('c:asset/c:unit', NS).get('meter'))
    assert root.find('c:asset/c:up_axis', NS).text == 'Z_UP'
    effects = {e.get('id'): e for e in root.findall('c:library_effects/c:effect', NS)}
    colors = {}
    for material in root.findall('c:library_materials/c:material', NS):
        effect = effects[material.find('c:instance_effect', NS).get('url')[1:]]
        color = effect.find('.//c:diffuse/c:color', NS)
        colors[material.get('id')] = vec(color.text).tolist() if color is not None else [.5, .5, .5, 1]
    geometries = {g.get('id'): g for g in root.findall('c:library_geometries/c:geometry', NS)}
    parts = []
    for node in root.findall('c:library_visual_scenes/c:visual_scene/c:node', NS):
        assert {c.tag.split('}')[-1] for c in node} <= {'matrix', 'instance_geometry'}
        matrix = node.find('c:matrix', NS)
        matrix = vec(matrix.text).reshape(4, 4) if matrix is not None else np.eye(4)
        assert np.allclose(matrix, np.eye(4)), 'Unexpected source node transform'
        inst = node.find('c:instance_geometry', NS)
        mesh = geometries[inst.get('url')[1:]].find('c:mesh', NS)
        materials = {m.get('symbol'): m.get('target')[1:] for m in inst.findall('.//c:instance_material', NS)}
        arrays = {}
        for source in mesh.findall('c:source', NS):
            accessor = source.find('c:technique_common/c:accessor', NS)
            arrays[source.get('id')] = vec(source.find('c:float_array', NS).text).reshape(-1, int(accessor.get('stride', '1')))
        vertices = {v.get('id'): v.find("c:input[@semantic='POSITION']", NS).get('source')[1:] for v in mesh.findall('c:vertices', NS)}
        for primitive in mesh:
            kind = primitive.tag.split('}')[-1]
            if kind in ('source', 'vertices', 'lines'):
                continue
            assert kind in ('triangles', 'polylist'), kind
            inputs = primitive.findall('c:input', NS)
            stride = max(int(i.get('offset', '0')) for i in inputs) + 1
            vertex_input = next(i for i in inputs if i.get('semantic') == 'VERTEX')
            points = arrays[vertices[vertex_input.get('source')[1:]]][:, :3] * unit
            indices = np.fromstring(primitive.find('c:p', NS).text, sep=' ', dtype=int).reshape(-1, stride)[:, int(vertex_input.get('offset', '0'))]
            if kind == 'polylist':
                assert np.all(np.fromstring(primitive.find('c:vcount', NS).text, sep=' ', dtype=int) == 3)
            indices = indices.reshape(-1, 3)
            if not len(indices):
                continue
            used, remapped = np.unique(indices, return_inverse=True)
            parts.append((points[used], remapped.reshape(-1, 3), colors[materials[primitive.get('material')]]))
    return parts


def obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith('v '): vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            tokens = line.split()[1:]
            assert len(tokens) == 3
            faces.append([int(t.split('/')[0]) - 1 for t in tokens])
    return np.asarray(vertices), np.asarray(faces)


def contact_face_center(side):
    """Area-weighted center of the actual inward planar rubber contact surface.

    The 2F-85 URDF has no pad/TCP frames. Its open fingertip meshes have planar
    inward faces at their X extrema: negative X for left, positive X for right.
    Use those faces, not the fingertip joint origins or the 2F-140 pad convention.
    """
    parts = collada_parts(GRIP / f'meshes/visual/2f_85/{side}_finger_tip.dae')
    all_points = np.vstack([p for p, _, _ in parts])
    plane = all_points[:, 0].min() if side == 'left' else all_points[:, 0].max()
    centers, areas = [], []
    for points, faces, _ in parts:
        triangles = points[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1) / 2
        selected = np.all(np.abs(triangles[:, :, 0] - plane) < 1e-7, axis=1) & (area > 1e-10)
        centers.extend(triangles[selected].mean(axis=1))
        areas.extend(area[selected])
    assert len(areas) == 20 and 0.0008 < sum(areas) < 0.0009, 'Unexpected 2F-85 contact face'
    return np.average(centers, weights=areas, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep pristine upstream xacro files; disable only controller integration in a build copy.
    macro = (GRIP / 'robotiq_2f_85_macro.urdf.xacro').read_text()
    macro = '\n'.join(line for line in macro.splitlines() if '2f_85.ros2_control.xacro' not in line)
    macro = macro.replace('$(find robotiq_description)/urdf/', str(GRIP) + '/')
    (GRIP / 'build_macro.xacro').write_text(macro)
    wrapper = f'''<robot name="robotiq_2f85" xmlns:xacro="http://www.ros.org/wiki/xacro">
      <link name="tool0"/>
      <xacro:include filename="{GRIP / 'ur_to_robotiq_adapter.urdf.xacro'}"/>
      <xacro:include filename="{GRIP / 'build_macro.xacro'}"/>
      <xacro:ur_to_robotiq prefix="" parent="tool0" child="gripper_mount_link" rotation="0"/>
      <xacro:robotiq_gripper name="gripper" prefix="" parent="gripper_mount_link" include_ros2_control="false">
        <origin xyz="0 0 0" rpy="0 0 0"/>
      </xacro:robotiq_gripper>
    </robot>'''
    (GRIP / 'wrapper.xacro').write_text(wrapper)
    gripper = ET.fromstring(xacro.process_file(str(GRIP / 'wrapper.xacro')).toxml())
    # TCP from the open mesh contact surfaces. Keep it rigid thereafter.
    poses = {'tool0': np.eye(4)}
    pending = list(gripper.findall('joint'))
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in poses]
        assert ready
        for j in ready:
            poses[j.find('child').get('link')] = poses[j.find('parent').get('link')] @ transform(j)
            pending.remove(j)
    base_T = poses['robotiq_85_base_link']
    contacts = {side: (np.linalg.inv(base_T) @ poses[f'robotiq_85_{side}_finger_tip_link'] @
                      np.r_[contact_face_center(side), 1])[:3] for side in ('left', 'right')}
    tcp_xyz = (contacts['left'] + contacts['right']) / 2
    # Plane separation along opening axis; source CAD has a small asymmetric Y offset.
    gap = abs(contacts['left'][0] - contacts['right'][0])
    ET.SubElement(gripper, 'link', name='classroom_tcp')
    j = ET.SubElement(gripper, 'joint', name='classroom_tcp_joint', type='fixed')
    ET.SubElement(j, 'parent', link='robotiq_85_base_link')
    ET.SubElement(j, 'child', link='classroom_tcp')
    ET.SubElement(j, 'origin', xyz=' '.join(map(str, tcp_xyz)), rpy='0 0 0')
    original_gripper_joints = []
    for j in gripper.findall('joint'):
        if j.get('type') != 'fixed':
            original_gripper_joints.append({'name': j.get('name'), 'type': j.get('type'), 'openAngle': 0.,
                                           'mimic': dict(j.find('mimic').attrib) if j.find('mimic') is not None else None})
            j.set('type', 'fixed')
            for node in list(j):
                if node.tag in ('axis', 'limit', 'dynamics', 'mimic'): j.remove(node)
    arm = ET.parse(UR / 'source.urdf').getroot()
    arm.set('name', 'ur3_cb3_robotiq_2f85_classroom')
    for node in list(arm):
        if node.tag not in ('link', 'joint'): arm.remove(node)
    wrist3 = next(j for j in arm.findall('joint') if j.get('name') == 'wrist_3_joint')
    assert wrist3.get('type') == 'continuous'
    wrist3.set('type', 'revolute')
    wrist3.find('limit').set('lower', str(-2*np.pi)); wrist3.find('limit').set('upper', str(2*np.pi))
    for node in gripper:
        if node.tag in ('link','joint') and not (node.tag == 'link' and node.get('name') == 'tool0'):
            arm.append(node)
    links = {l.get('name'): {'name': l.get('name'), 'visuals': []} for l in arm.findall('link')}
    blob = bytearray(); meshes = {}; source_hashes = {}
    for filename in ('source.urdf', 'source_wrapper.xacro', 'robot.xml', 'LICENSE'):
        source_hashes['ur3/' + filename] = hashlib.sha256((UR / filename).read_bytes()).hexdigest()
    for filename in ('robotiq_2f_85_macro.urdf.xacro', 'ur_to_robotiq_adapter.urdf.xacro', 'LICENSE'):
        source_hashes['robotiq_source/' + filename] = hashlib.sha256((GRIP / filename).read_bytes()).hexdigest()
    def add_mesh(name, points, faces):
        assert points.shape[1] == faces.shape[1] == 3
        assert np.isfinite(points).all() and np.min(faces) >= 0 and np.max(faces) < len(points)
        meshes[name] = {'vertexOffset':len(blob), 'vertexCount':len(points)}
        blob.extend(points.astype('<f4').tobytes())
        meshes[name].update(indexOffset=len(blob), indexCount=int(faces.size))
        blob.extend(faces.astype('<u4').tobytes())
    # Existing local UR3 parts have material split and preserve original visual pose.
    mjcf = ET.parse(UR / 'robot.xml').getroot()
    mesh_files = {m.get('name'): m.get('file') for m in mjcf.findall('asset/mesh')}
    for body in mjcf.findall('.//body'):
        for g in body.findall('geom'):
            if g.get('group') != '2': continue
            key = 'ur3_' + g.get('mesh'); path = UR / mesh_files[g.get('mesh')]
            points, faces = obj(path); add_mesh(key, points, faces)
            source_hashes[str(path.relative_to(UR))] = hashlib.sha256(path.read_bytes()).hexdigest()
            quat = vec(g.get('quat','1 0 0 0'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rpy = Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz').tolist()
            links[body.get('name')]['visuals'].append({'xyz':vec(g.get('pos','0 0 0')).tolist(), 'rpy':rpy,
                                                      'scale':[1,1,1], 'rgba':vec(g.get('rgba')).tolist(), 'mesh':key})
    for link in gripper.findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            path = GRIP / mesh.get('filename').removeprefix('package://robotiq_description/')
            source_hashes['robotiq_source/' + str(path.relative_to(GRIP))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for n, (points, faces, color) in enumerate(collada_parts(path)):
                key = link.get('name') + '_' + str(n); add_mesh(key, points, faces)
                links[link.get('name')]['visuals'].append({**origin(visual.find('origin')), 'scale':vec(mesh.get('scale','1 1 1')).tolist(), 'rgba':color, 'mesh':key})
    joints = []
    for joint in arm.findall('joint'):
        item = {**origin(joint.find('origin')), 'name':joint.get('name'), 'type':joint.get('type'),
                'parent':joint.find('parent').get('link'), 'child':joint.find('child').get('link'),
                'axis':vec(joint.find('axis').get('xyz') if joint.find('axis') is not None else '0 0 1').tolist()}
        if joint.get('type') != 'fixed':
            item.update(lower=float(joint.find('limit').get('lower')),upper=float(joint.find('limit').get('upper')))
        joints.append(item)
    assert len({j['child'] for j in joints}) == len(joints), 'A child must have exactly one parent'
    assert set(links) - {j['child'] for j in joints} == {'world'}
    assert len(links) == len(joints) + 1
    model = {'name':'UR3 CB3 + Robotiq 2F-85', 'rootFrame':'world', 'tcpFrame':'classroom_tcp',
             'meshes':meshes, 'links':list(links.values()), 'joints':joints,
             'gripper':{'model':'Robotiq 2F-85', 'posture':'fully open; fixed', 'padGapMetres':gap}}
    # Traverse the one chain world -> TCP. Fixed roots fold into first arm origin.
    parents = {j.find('child').get('link'):j for j in arm.findall('joint')}
    path = []; child = 'classroom_tcp'
    while child in parents:
        j = parents[child]; path.append(j); child = j.find('parent').get('link')
    path.reverse(); T = np.eye(4); active = []
    for j in path:
        T = T @ transform(j)
        if j.get('name') in ARM:
            active.append({'name':j.get('name'),'xyz':T[:3,3].tolist(),'rotation':T[:3,:3].tolist(),
                           'axis':vec(j.find('axis').get('xyz')).tolist(),
                           'lower':float(j.find('limit').get('lower')),'upper':float(j.find('limit').get('upper'))})
            T = np.eye(4)
    assert [j['name'] for j in active] == ARM
    kinematics = {'joints':active, 'tool':{'xyz':T[:3,3].tolist(),'rotation':T[:3,:3].tolist()},
                  'rootFrame':'world','tcpFrame':'classroom_tcp','characteristicLength':0.4,
                  'wrist3Note':'Continuous physical joint; [-2pi,2pi] is only the classroom numerical search interval.'}
    numerical = copy.deepcopy(arm)
    for link in numerical.findall('link'):
        for tag in ('visual','collision'):
            for element in list(link.findall(tag)): link.remove(element)
    ET.indent(numerical)
    numerical_xml = ET.tostring(numerical, encoding='unicode') + '\n'
    (MODELS / 'ur3_2f85.urdf').write_text(numerical_xml)
    (ROOT / 'dist/models').mkdir(parents=True, exist_ok=True)
    (ROOT / 'dist/models/ur3_2f85.urdf').write_text(numerical_xml)
    (OUT / 'robot.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    (OUT / 'kinematics.json').write_text(json.dumps(kinematics,indent=2)+'\n')
    (OUT / 'meshes.bin').write_bytes(blob)
    (OUT / 'UR_LICENSE.txt').write_text((UR/'LICENSE').read_text())
    (OUT / 'ROBOTIQ_LICENSE.txt').write_text((GRIP/'LICENSE').read_text())
    provenance = {'ur3':json.loads((UR/'provenance.json').read_text()),
                  'robotiq':{'repository':'https://github.com/robotiq/ros','commit':COMMIT,
                             'directory':'grippers/robotiq_description','fixedJoints':original_gripper_joints},
                  'mounting':'Official ur_to_robotiq adapter on tool0, zero yaw parameter, +0.011 m. Official 2F85 base has zero yaw relative to the adapter; opening direction is base X. All gripper and adapter meshes come from 2f_85/.',
                  'tcp':'Uncalibrated classroom frame: fixed at midpoint of area-weighted centers of the two inward open fingertip contact faces, oriented with robotiq_85_base_link. Contact faces are extracted from the official visual meshes; upstream has no pad/TCP frames.',
                  'tcpInGripperBase':tcp_xyz.tolist(),
                  'contactFaceCentersInGripperBase':{side:p.tolist() for side,p in contacts.items()},
                  'padGapMetres':gap,'padGapDefinition':'Distance between the two inward contact planes along gripper-base X; source meshes have small asymmetric Y offsets.',
                  'searchIntervalNote':kinematics['wrist3Note'],
                  'source_sha256':source_hashes,'binarySha256':hashlib.sha256(blob).hexdigest()}
    (OUT / 'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (MODELS / 'README.txt').write_text('UR3 CB3, not UR3e. Six numerical revolute joints; wrist_3_joint is physically continuous, with a classroom search interval of [-2pi,2pi].\nRobotiq 2F-85 is rigidly mounted through its official 11 mm adapter and fixed fully open.\nclassroom_tcp is the midpoint of area-weighted centers of the two inward fully open fingertip contact faces, extracted from official visual meshes and oriented with the gripper base. Upstream has no pad/TCP frames. It is an uncalibrated classroom convention.\nThis URDF omits visual/collision geometry and is for numerical verification. Browser geometry, source attribution and licenses are in dist/assets/robot/.\n')
    print(json.dumps({'meshes':len(meshes),'triangles':sum(m['indexCount']//3 for m in meshes.values()),'binaryBytes':len(blob),'tcpBase':tcp_xyz.tolist(),'tool':kinematics['tool'],'padGap':gap},indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ur-source', type=Path, required=True,
                        help='Prepared UR3 source directory: source.urdf, robot.xml, material-split OBJ meshes, provenance.json, LICENSE')
    UR = parser.parse_args().ur_source.resolve()
    main()
