/**
 * Three.js renderer for the kinematic avatar — the web counterpart of
 * avatar/gl/AvatarGLRenderer.kt and AvatarMeshBuilder.kt.
 *
 * This is the one part of the avatar that is a rewrite rather than a port: the
 * Android side is hand-written OpenGL ES 2.0 with its own shader, mesh builder
 * and orbit controller, none of which transfers. What IS ported exactly is
 * everything visible — the colours, radii, offsets and camera, all transcribed
 * from the Kotlin so the two builds render the same figure.
 *
 * Geometry is allocated once and only transformed per frame. A sign moves ~100
 * primitives every frame (42 finger joints, 40 finger bones, plus the body), so
 * rebuilding meshes each frame would dominate the budget.
 */

import * as THREE from 'three'

import { HAND_BONES } from './kinematics'
import type { AvatarPoseState } from './motionPlayer'
import type { Vec3 } from './motion'

// Transcribed from AvatarGLRenderer.kt. The comments there name them.
const COLOR_SKIN = 0xaa866a // natural warm mannequin tone
const COLOR_EYE = 0x141414 // solid black dot eyes
const COLOR_BODY = 0x0b4ea3 // royal blue body
const COLOR_GRID = 0x1f2430 // subtle studio floor grid
const COLOR_BACKGROUND = 0x090d13 // studio dark

/** Radii, in metres, exactly as the Kotlin draws them. */
const R = {
  head: 0.13,
  eye: 0.0085,
  neck: 0.044,
  torso: 0.165,
  shoulder: 0.046,
  upperArm: 0.038,
  elbow: 0.038,
  forearm: 0.032,
  wrist: 0.026,
  fingerJointA: 0.007,
  fingerJointB: 0.0065,
  fingerBone: 0.0055,
}

const EYE_SPACING = 0.038
const EYE_Y_OFFSET = 0.01
const EYE_Z_OFFSET = 0.124
const HEAD_Y = 1.33

/** Reused across every per-frame transform so the loop allocates nothing. */
const _a = new THREE.Vector3()
const _b = new THREE.Vector3()
const _dir = new THREE.Vector3()
const _mid = new THREE.Vector3()
const _quat = new THREE.Quaternion()
const UP = new THREE.Vector3(0, 1, 0)

/**
 * Orients and stretches a unit-height cylinder to span two points.
 *
 * Three's CylinderGeometry is Y-up and centred, so spanning an arbitrary
 * segment means positioning at the midpoint, rotating Y onto the segment
 * direction, and scaling Y by its length.
 */
function spanCylinder(mesh: THREE.Mesh, from: Vec3, to: Vec3, radius: number): void {
  _a.set(from.x, from.y, from.z)
  _b.set(to.x, to.y, to.z)
  _dir.subVectors(_b, _a)
  const len = _dir.length()

  if (len < 1e-6) {
    mesh.visible = false
    return
  }
  mesh.visible = true

  _mid.addVectors(_a, _b).multiplyScalar(0.5)
  mesh.position.copy(_mid)
  _quat.setFromUnitVectors(UP, _dir.normalize())
  mesh.quaternion.copy(_quat)
  mesh.scale.set(radius, len, radius)
}

function sphereAt(mesh: THREE.Mesh, p: Vec3, radius: number): void {
  mesh.visible = true
  mesh.position.set(p.x, p.y, p.z)
  mesh.scale.setScalar(radius)
}

export class AvatarRenderer {
  private scene = new THREE.Scene()
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer

  // Unit primitives, shared by every part. Scaling handles the rest.
  private sphereGeo = new THREE.SphereGeometry(1, 24, 24)
  private cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 24)

  private skin: THREE.Material
  private body: THREE.Material
  private eyeMat: THREE.Material

  // Body parts, allocated once.
  private head!: THREE.Mesh
  private eyeL!: THREE.Mesh
  private eyeR!: THREE.Mesh
  private neck!: THREE.Mesh
  private torso!: THREE.Mesh
  private shoulderL!: THREE.Mesh
  private shoulderR!: THREE.Mesh
  private upperArmL!: THREE.Mesh
  private upperArmR!: THREE.Mesh
  private elbowL!: THREE.Mesh
  private elbowR!: THREE.Mesh
  private forearmL!: THREE.Mesh
  private forearmR!: THREE.Mesh
  private wristL!: THREE.Mesh
  private wristR!: THREE.Mesh

  // Finger pools: 21 joints and 20 bones per hand.
  private jointsL: THREE.Mesh[] = []
  private jointsR: THREE.Mesh[] = []
  private bonesL: THREE.Mesh[] = []
  private bonesR: THREE.Mesh[] = []

  /** Orbit state, mirroring TouchOrbitController.kt. */
  private azimuth = 0
  private elevation = 0.12
  private distance = 2.1
  private readonly target = new THREE.Vector3(0, 1.05, 0)

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.scene.background = new THREE.Color(COLOR_BACKGROUND)

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)

    // A directional key light at the Kotlin's light position, plus enough
    // ambient that the unlit side stays readable rather than black.
    const key = new THREE.DirectionalLight(0xffffff, 2.2)
    key.position.set(2.5, 4.5, 3.5)
    this.scene.add(key)
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55))

    this.skin = new THREE.MeshLambertMaterial({ color: COLOR_SKIN })
    this.body = new THREE.MeshLambertMaterial({ color: COLOR_BODY })
    this.eyeMat = new THREE.MeshBasicMaterial({ color: COLOR_EYE })

    this.buildFloor()
    this.buildBody()
    this.buildHands()
    this.resize()
  }

  private mesh(geo: THREE.BufferGeometry, mat: THREE.Material): THREE.Mesh {
    const m = new THREE.Mesh(geo, mat)
    this.scene.add(m)
    return m
  }

  private buildFloor(): void {
    const grid = new THREE.GridHelper(6, 24, COLOR_GRID, COLOR_GRID)
    const mat = grid.material as THREE.Material
    mat.transparent = true
    mat.opacity = 0.5
    this.scene.add(grid)
  }

  private buildBody(): void {
    this.head = this.mesh(this.sphereGeo, this.skin)
    this.eyeL = this.mesh(this.sphereGeo, this.eyeMat)
    this.eyeR = this.mesh(this.sphereGeo, this.eyeMat)
    this.neck = this.mesh(this.cylinderGeo, this.skin)
    this.torso = this.mesh(this.cylinderGeo, this.body)
    this.shoulderL = this.mesh(this.sphereGeo, this.body)
    this.shoulderR = this.mesh(this.sphereGeo, this.body)
    this.upperArmL = this.mesh(this.cylinderGeo, this.body)
    this.upperArmR = this.mesh(this.cylinderGeo, this.body)
    this.elbowL = this.mesh(this.sphereGeo, this.body)
    this.elbowR = this.mesh(this.sphereGeo, this.body)
    this.forearmL = this.mesh(this.cylinderGeo, this.skin)
    this.forearmR = this.mesh(this.cylinderGeo, this.skin)
    this.wristL = this.mesh(this.sphereGeo, this.skin)
    this.wristR = this.mesh(this.sphereGeo, this.skin)
  }

  private buildHands(): void {
    for (const pool of [this.jointsL, this.jointsR]) {
      for (let i = 0; i < 21; i++) pool.push(this.mesh(this.sphereGeo, this.skin))
    }
    for (const pool of [this.bonesL, this.bonesR]) {
      for (let i = 0; i < HAND_BONES.length; i++) pool.push(this.mesh(this.cylinderGeo, this.skin))
    }
  }

  /** Orbit input, in radians. Elevation is clamped to keep the floor sensible. */
  orbit(deltaAzimuth: number, deltaElevation: number): void {
    this.azimuth += deltaAzimuth
    this.elevation = Math.min(Math.max(this.elevation + deltaElevation, -0.6), 1.2)
  }

  zoom(factor: number): void {
    this.distance = Math.min(Math.max(this.distance * factor, 0.9), 5)
  }

  /** Preset camera angles, matching the Android inspection buttons. */
  setView(view: 'front' | 'side45' | 'profile90' | 'behind' | 'hands'): void {
    const presets = {
      front: { az: 0, el: 0.12, d: 2.1, ty: 1.05 },
      side45: { az: Math.PI / 4, el: 0.12, d: 2.1, ty: 1.05 },
      profile90: { az: Math.PI / 2, el: 0.1, d: 2.1, ty: 1.05 },
      behind: { az: Math.PI, el: 0.12, d: 2.1, ty: 1.05 },
      hands: { az: 0, el: 0.05, d: 1.0, ty: 0.95 },
    }
    const p = presets[view]
    this.azimuth = p.az
    this.elevation = p.el
    this.distance = p.d
    this.target.set(0, p.ty, 0)
  }

  resize(): void {
    const w = this.canvas.clientWidth || 1
    const h = this.canvas.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  render(pose: AvatarPoseState): void {
    this.applyPose(pose)

    const cosEl = Math.cos(this.elevation)
    this.camera.position.set(
      this.target.x + this.distance * cosEl * Math.sin(this.azimuth),
      this.target.y + this.distance * Math.sin(this.elevation),
      this.target.z + this.distance * cosEl * Math.cos(this.azimuth),
    )
    this.camera.lookAt(this.target)
    this.renderer.render(this.scene, this.camera)
  }

  private applyPose(pose: AvatarPoseState): void {
    // Head sits at a fixed height; only its x tracks the signer. The non-uniform
    // scale is what makes it an oval rather than a ball.
    const h = { x: pose.headPosition.x, y: HEAD_Y, z: pose.headPosition.z }
    this.head.position.set(h.x, h.y, h.z)
    this.head.scale.set(R.head * 1.0, R.head * 1.15, R.head * 0.98)

    const eyeY = h.y + EYE_Y_OFFSET
    const eyeZ = h.z + EYE_Z_OFFSET
    sphereAt(this.eyeL, { x: h.x + EYE_SPACING, y: eyeY, z: eyeZ }, R.eye)
    sphereAt(this.eyeR, { x: h.x - EYE_SPACING, y: eyeY, z: eyeZ }, R.eye)

    spanCylinder(this.neck, { x: 0, y: 1.15, z: 0 }, { x: 0, y: 1.22, z: 0 }, R.neck)
    spanCylinder(this.torso, { x: 0, y: 0.65, z: 0 }, { x: 0, y: 1.15, z: 0 }, R.torso)

    sphereAt(this.shoulderL, pose.leftShoulder, R.shoulder)
    sphereAt(this.shoulderR, pose.rightShoulder, R.shoulder)

    spanCylinder(this.upperArmL, pose.leftShoulder, pose.leftElbow, R.upperArm)
    spanCylinder(this.upperArmR, pose.rightShoulder, pose.rightElbow, R.upperArm)
    sphereAt(this.elbowL, pose.leftElbow, R.elbow)
    sphereAt(this.elbowR, pose.rightElbow, R.elbow)
    spanCylinder(this.forearmL, pose.leftElbow, pose.leftWrist, R.forearm)
    spanCylinder(this.forearmR, pose.rightElbow, pose.rightWrist, R.forearm)
    sphereAt(this.wristL, pose.leftWrist, R.wrist)
    sphereAt(this.wristR, pose.rightWrist, R.wrist)

    this.applyHand(pose.leftHandPoints, this.jointsL, this.bonesL)
    this.applyHand(pose.rightHandPoints, this.jointsR, this.bonesR)
  }

  private applyHand(
    points: Vec3[] | null,
    joints: THREE.Mesh[],
    bones: THREE.Mesh[],
  ): void {
    // A hand the capture did not see is hidden outright rather than frozen at
    // its last position, which would read as a detached hand hanging in space.
    if (!points || points.length !== 21) {
      joints.forEach((m) => (m.visible = false))
      bones.forEach((m) => (m.visible = false))
      return
    }

    for (let i = 0; i < 21; i++) {
      sphereAt(joints[i], points[i], i === 0 ? R.fingerJointA : R.fingerJointB)
    }
    HAND_BONES.forEach(([a, b], i) => {
      spanCylinder(bones[i], points[a], points[b], R.fingerBone)
    })
  }

  dispose(): void {
    this.sphereGeo.dispose()
    this.cylinderGeo.dispose()
    this.skin.dispose()
    this.body.dispose()
    this.eyeMat.dispose()
    this.renderer.dispose()
  }
}

// Re-exported so the screen can size the canvas without importing three itself.
export { THREE }
