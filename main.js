import * as THREE from 'three';
import './style.css'; // Import the CSS file

let scene, camera, renderer, mesh, uniforms;
const clock = new THREE.Clock();

// Vertex Shader: Passes UV coordinates to the fragment shader
// and positions the vertices.
const vertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// Fragment Shader: Creates a vibrant, animated pattern.
// Replace this with your custom shader code!
const fragmentShader = `
    varying vec2 vUv; // UV coordinates (0 to 1 across the plane)
    uniform float u_time; // Time uniform for animations
    uniform vec2 u_resolution; // Resolution of the canvas (width, height)
    uniform vec3 u_primaryColor;
    uniform vec3 u_secondaryColor;
    uniform float u_animationSpeed;
    uniform float u_seed;

    // Helper function to create a wave pattern
    float wave(vec2 st, float freq, float offset, float time_multiplier) {
        return 0.5 + 0.5 * sin(st.x * freq + u_time * u_animationSpeed * time_multiplier + cos(st.y * freq * 0.8 + u_time * u_animationSpeed * time_multiplier * 0.7 + offset) * 2.0);
    }

    void main() {
        vec2 st = vUv;
        st.x *= u_resolution.x / u_resolution.y; // Aspect ratio correction

        float pattern = 0.0;
        // Use u_seed to vary parameters.
        // Multiplying by small numbers and using sin/cos keeps changes somewhat bounded.
        float s1 = sin(u_seed * 0.1); float c1 = cos(u_seed * 0.15);
        float s2 = sin(u_seed * 0.2); float c2 = cos(u_seed * 0.25);
        float s3 = sin(u_seed * 0.3); float c3 = cos(u_seed * 0.35);

        // Combine multiple wave patterns for complexity
        pattern += wave(st + c1*0.1, 8.0 + s1*2.0, 0.0 + c2*0.5, 0.5);
        pattern += wave(st * (1.5 + s2*0.2) + (0.2 + c1*0.1), 6.0 + c2*1.5, 1.0 + s3*0.8, 0.3);
        pattern += wave(st * (0.8 + c3*0.1) - (0.1 + s1*0.05), 10.0 + s2*2.5, 2.0 + c1*1.2, 0.4);
        pattern = mod(pattern, 1.0); // Keep pattern in 0-1 range

        // Mix between primary and secondary colors based on the pattern
        vec3 color = mix(u_primaryColor, u_secondaryColor, pattern);

        gl_FragColor = vec4(color, 1.0);
    }
`;

function init() {
    // Get the canvas element
    const canvas = document.getElementById('shaderCanvas');

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true // Enable antialiasing for smoother edges
    });
    renderer.setPixelRatio(window.devicePixelRatio); // Adjust for high DPI screens
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Scene
    scene = new THREE.Scene();

    // Camera: Orthographic camera for 2D rendering.
    // These settings mean the camera views a 2x2 unit square centered at the origin.
    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 1; // Position the camera to look at the plane

    // Initial control values
    const initialPrimaryColor = new THREE.Color(document.getElementById('primaryColor').value);
    const initialSecondaryColor = new THREE.Color(document.getElementById('secondaryColor').value);
    const initialAnimationSpeed = parseFloat(document.getElementById('animationSpeed').value);
    const initialSeed = Math.random() * 100.0; // Initial random seed

    // Uniforms for the shader
    uniforms = {
        u_time: { value: 0.0 },
        // u_resolution will store the actual pixel dimensions of the canvas
        u_resolution: { value: new THREE.Vector2(renderer.domElement.width, renderer.domElement.height) },
        u_primaryColor: { value: new THREE.Vector3(initialPrimaryColor.r, initialPrimaryColor.g, initialPrimaryColor.b) },
        u_secondaryColor: { value: new THREE.Vector3(initialSecondaryColor.r, initialSecondaryColor.g, initialSecondaryColor.b) },
        u_animationSpeed: { value: initialAnimationSpeed },
        u_seed: { value: initialSeed }
    };

    // Geometry: A plane that fills the camera's view.
    // A 2x2 plane perfectly matches the OrthographicCamera's view.
    const geometry = new THREE.PlaneGeometry(2, 2);

    // Material: Use ShaderMaterial to use our custom shaders.
    const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader
    });

    // Mesh: Combine geometry and material to create a mesh.
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh); // Add the mesh to the scene

    // Event Listeners
    window.addEventListener('resize', onWindowResize, false);
    setupUIListeners();
}

function onWindowResize() {
    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;

    // Update renderer size
    renderer.setSize(newWidth, newHeight);

    // Update u_resolution uniform with the new renderer dimensions (actual buffer size)
    uniforms.u_resolution.value.set(renderer.domElement.width, renderer.domElement.height);

    // Note: The OrthographicCamera with fixed left/right/top/bottom values
    // doesn't need its projection matrix updated on resize if the goal is
    // to always map the 2x2 plane to the full canvas. The viewport
    // transformation handles the aspect ratio.
}

function setupUIListeners() {
    const menuToggle = document.getElementById('menuToggle');
    const controlPanel = document.getElementById('controlPanel');
    const primaryColorInput = document.getElementById('primaryColor');
    const secondaryColorInput = document.getElementById('secondaryColor');
    const animationSpeedInput = document.getElementById('animationSpeed');
    const speedValueDisplay = document.getElementById('speedValue');
    const randomizeButton = document.getElementById('randomizeButton');

    menuToggle.addEventListener('click', () => {
        controlPanel.classList.toggle('open');
    });

    primaryColorInput.addEventListener('input', (event) => {
        const color = new THREE.Color(event.target.value);
        uniforms.u_primaryColor.value.set(color.r, color.g, color.b);
    });

    secondaryColorInput.addEventListener('input', (event) => {
        const color = new THREE.Color(event.target.value);
        uniforms.u_secondaryColor.value.set(color.r, color.g, color.b);
    });

    animationSpeedInput.addEventListener('input', (event) => {
        const speed = parseFloat(event.target.value);
        uniforms.u_animationSpeed.value = speed;
        speedValueDisplay.textContent = speed.toFixed(1);
    });

    randomizeButton.addEventListener('click', () => {
        uniforms.u_seed.value = Math.random() * 100.0; // Set a new random seed
    });
}

function animate() {
    requestAnimationFrame(animate); // Request the next frame

    // Update uniforms
    uniforms.u_time.value = clock.getElapsedTime(); // Pass current time to shader

    // Render the scene
    renderer.render(scene, camera);
}

// Initialize and start the animation loop
try {
    init();
    animate();
} catch (error) {
    console.error("An error occurred during Three.js initialization:", error);
    document.body.innerHTML = "<div style='color: white; text-align: center; padding-top: 50px; font-family: sans-serif;'>Sorry, an error occurred. WebGL might not be supported or enabled in your browser.</div>";
}