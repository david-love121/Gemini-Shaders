import * as THREE from 'three';
import './style.css'; // Import the CSS file

let scene, camera, renderer, mesh, uniforms;
const clock = new THREE.Clock();
let currentMode = '2d';
let currentSimulation = 'raymarching';
let currentFractalType = 'mandelbulb';
let isColorCycling = false;
let colorCycleSpeed = 0.5;

// Performance optimization variables
let needsResize = false;
let needsMeshUpdate = false;
let lastCameraUpdate = 0;
const CAMERA_UPDATE_THROTTLE = 16; // 60fps throttling

// Camera controls for 3D mode
let cameraControls = {
    position: new THREE.Vector3(0, 0, 5),
    target: new THREE.Vector3(0, 0, 0),
    radius: 5,
    azimuth: 0, // horizontal rotation
    elevation: 0, // vertical rotation
    keys: {
        w: false, a: false, s: false, d: false,
        up: false, down: false, left: false, right: false
    },
    mouse: {
        isDown: false,
        lastX: 0,
        lastY: 0
    }
};

// 2D Vertex Shader: Passes UV coordinates to the fragment shader
const vertexShader2D = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// 2D Fragment Shader: Creates a vibrant, animated pattern
const fragmentShader2D = `
    varying vec2 vUv;
    uniform float u_time;
    uniform vec2 u_resolution;
    uniform vec3 u_primaryColor;
    uniform vec3 u_secondaryColor;
    uniform float u_animationSpeed;
    uniform float u_seed;
    uniform float u_bloomIntensity;
    uniform float u_metallicness;
    uniform float u_fractalType;
    uniform float u_evolutionPhase;

    // Optimized wave function with cached calculations
    float wave(vec2 st, float freq, float offset, float time_multiplier) {
        float timeComponent = u_time * u_animationSpeed * time_multiplier;
        float yComponent = st.y * freq * 0.8 + timeComponent * 0.7 + offset;
        return 0.5 + 0.5 * sin(st.x * freq + timeComponent + cos(yComponent) * 2.0);
    }

    void main() {
        vec2 st = vUv;
        st.x *= u_resolution.x / u_resolution.y;

        // Cache seed-based calculations
        float s1 = sin(u_seed * 0.1), c1 = cos(u_seed * 0.15);
        float s2 = sin(u_seed * 0.2), c2 = cos(u_seed * 0.25);
        float s3 = sin(u_seed * 0.3), c3 = cos(u_seed * 0.35);

        // Optimized pattern calculation with reduced redundancy
        vec2 offset1 = vec2(c1 * 0.1, 0.0);
        vec2 offset2 = vec2(0.2 + c1 * 0.1, 0.0);
        vec2 offset3 = vec2(-0.1 - s1 * 0.05, 0.0);
        
        float pattern = wave(st + offset1, 8.0 + s1 * 2.0, c2 * 0.5, 0.5) +
                       wave(st * (1.5 + s2 * 0.2) + offset2, 6.0 + c2 * 1.5, 1.0 + s3 * 0.8, 0.3) +
                       wave(st * (0.8 + c3 * 0.1) + offset3, 10.0 + s2 * 2.5, 2.0 + c1 * 1.2, 0.4);
        
        pattern = mod(pattern, 1.0);

        vec3 color = mix(u_primaryColor, u_secondaryColor, pattern);
        
        // Optimized metallic shimmer
        float shimmerPhase = pattern * 10.0 + u_time * u_animationSpeed * 2.0;
        float shimmer = sin(shimmerPhase) * 0.5 + 0.5;
        color = mix(color, color * 1.5, shimmer * u_metallicness * 0.3);
        
        // Apply bloom with early exit
        float brightness = dot(color, vec3(0.299, 0.587, 0.114));
        if(brightness > 0.5) {
            color += (brightness - 0.5) * u_bloomIntensity * vec3(1.0, 0.9, 0.8);
        }
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

// 3D Vertex Shader
const vertexShader3D = `
    varying vec3 vPosition;
    varying vec2 vUv;
    
    void main() {
        vPosition = position;
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// 3D Fragment Shaders for different simulations
const get3DFragmentShader = (simulationType) => {
    const baseUniforms = `
        varying vec3 vPosition;
        varying vec2 vUv;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform vec3 u_primaryColor;
        uniform vec3 u_secondaryColor;
        uniform float u_animationSpeed;
        uniform float u_seed;
        uniform vec3 u_cameraPosition;
        uniform vec3 u_cameraTarget;
        uniform float u_bloomIntensity;
        uniform float u_metallicness;
        uniform float u_fractalType;
        uniform float u_evolutionPhase;
    `;

    switch(simulationType) {
        case 'raymarching':
            return baseUniforms + `
                // Advanced material functions
                vec3 hue2rgb(float h) {
                    h = fract(h);
                    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
                }
                
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                // Smooth minimum function for better blending
                float smin(float a, float b, float k) {
                    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
                    return mix(b, a, h) - k * h * (1.0 - h);
                }
                
                // Enhanced SDF primitives
                float sdSphere(vec3 p, float r) {
                    return length(p) - r;
                }
                
                float sdTorus(vec3 p, vec2 t) {
                    vec2 q = vec2(length(p.xz) - t.x, p.y);
                    return length(q) - t.y;
                }
                
                float sdOctahedron(vec3 p, float s) {
                    p = abs(p);
                    return (p.x + p.y + p.z - s) * 0.57735027;
                }
                
                // 3D noise for texturing
                vec3 hash3(vec3 p) {
                    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                             dot(p, vec3(269.5, 183.3, 246.1)),
                             dot(p, vec3(113.5, 271.9, 124.6)));
                    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
                }
                
                float noise(vec3 p) {
                    vec3 i = floor(p);
                    vec3 f = fract(p);
                    vec3 u = f * f * (3.0 - 2.0 * f);
                    
                    return mix(mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                                       dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                                       dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
                               mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                                       dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                                       dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
                }
                
                // Advanced scene with multiple materials
                vec2 map(vec3 p) {
                    vec3 originalP = p;
                    
                    // Global rotation with seed variation
                    float rotationSpeed = 0.3 + sin(u_seed * 0.1) * 0.2;
                    p.xz = mat2(cos(u_time * u_animationSpeed * rotationSpeed), -sin(u_time * u_animationSpeed * rotationSpeed),
                                sin(u_time * u_animationSpeed * rotationSpeed), cos(u_time * u_animationSpeed * rotationSpeed)) * p.xz;
                    
                    // Central crystalline structure
                    float mainSize = 1.2 + 0.4 * sin(u_time * u_animationSpeed * 0.7) + cos(u_seed * 0.05) * 0.3;
                    float crystal = sdOctahedron(p, mainSize);
                    
                    // Add surface detail to crystal
                    float surfaceNoise = noise(originalP * 8.0 + u_time * u_animationSpeed * 0.1) * 0.05;
                    crystal += surfaceNoise;
                    
                    float scene = crystal;
                    float materialID = 1.0; // Crystal material
                    
                    // Orbiting metallic spheres
                    int sphereCount = 5 + int(mod(u_seed * 0.1, 3.0));
                    for(int i = 0; i < 8; i++) {
                        if(i >= sphereCount) break;
                        
                        float fi = float(i);
                        float angleOffset = u_seed * 0.2 + fi * 1.256; // 72 degrees apart
                        float angle = angleOffset + u_time * u_animationSpeed * (0.8 + sin(u_seed * 0.05 + fi) * 0.4);
                        float radius = 2.5 + 0.8 * sin(u_time * u_animationSpeed * 0.4 + fi) + cos(u_seed * 0.03 + fi) * 0.5;
                        
                        vec3 orbitPos = vec3(
                            cos(angle) * radius,
                            sin(u_time * u_animationSpeed * 0.6 + fi * 0.7 + u_seed * 0.1) * 1.2,
                            sin(angle) * radius
                        );
                        
                        float sphereSize = 0.4 + 0.2 * sin(u_time * u_animationSpeed * 1.5 + fi * 2.5 + u_seed * 0.07);
                        float sphere = sdSphere(p - orbitPos, sphereSize);
                        
                        // Add metallic surface texture
                        float metalNoise = noise(originalP * 12.0 + u_time * u_animationSpeed * 0.05) * 0.02;
                        sphere += metalNoise;
                        
                        if(sphere < scene) {
                            scene = sphere;
                            materialID = 2.0; // Metallic material
                        }
                    }
                    
                    // Floating energy toroids
                    int torusCount = 3 + int(mod(u_seed * 0.15, 3.0));
                    for(int i = 0; i < 6; i++) {
                        if(i >= torusCount) break;
                        
                        float fi = float(i);
                        float torusAngle = u_time * u_animationSpeed * 0.3 + fi * 2.094; // 120 degrees
                        vec3 torusPos = vec3(
                            cos(torusAngle) * 4.0,
                            sin(u_time * u_animationSpeed * 0.25 + fi * 1.5) * 2.0,
                            sin(torusAngle) * 4.0
                        );
                        
                        vec3 localP = p - torusPos;
                        // Rotate torus
                        float torusRot = u_time * u_animationSpeed + fi * 1.57;
                        localP.xy = mat2(cos(torusRot), -sin(torusRot), sin(torusRot), cos(torusRot)) * localP.xy;
                        
                        float torus = sdTorus(localP, vec2(0.8, 0.2));
                        
                        // Add energy glow effect
                        float energyNoise = noise(originalP * 6.0 + u_time * u_animationSpeed * 0.3) * 0.1;
                        torus += energyNoise;
                        
                        if(torus < scene) {
                            scene = torus;
                            materialID = 3.0; // Energy material
                        }
                    }
                    
                    return vec2(scene, materialID);
                }
                
                // Enhanced normal calculation
                vec3 getNormal(vec3 p) {
                    float eps = 0.002;
                    return normalize(vec3(
                        map(p + vec3(eps, 0, 0)).x - map(p - vec3(eps, 0, 0)).x,
                        map(p + vec3(0, eps, 0)).x - map(p - vec3(0, eps, 0)).x,
                        map(p + vec3(0, 0, eps)).x - map(p - vec3(0, 0, eps)).x
                    ));
                }
                
                // Soft shadows with reduced iterations
                float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
                    float shadow = 1.0;
                    for(float t = mint; t < maxt;) {
                        float h = map(ro + rd * t).x;
                        if(h < 0.002) return 0.0; // Early exit
                        shadow = min(shadow, k * h / t);
                        t += max(h, 0.02); // Minimum step to prevent stalling
                        if(t > maxt) break;
                    }
                    return shadow;
                }
                
                // Optimized ambient occlusion with fewer samples
                float ambientOcclusion(vec3 p, vec3 n) {
                    float occ = 0.0;
                    float sca = 1.0;
                    for(int i = 0; i < 4; i++) { // Reduced from 5
                        float h = 0.02 + 0.15 * float(i) / 3.0;
                        float d = map(p + h * n).x;
                        occ += (h - d) * sca;
                        sca *= 0.9;
                    }
                    return clamp(1.0 - 1.8 * occ, 0.0, 1.0);
                }
                
                // Material-based lighting
                vec3 getMaterialColor(vec3 p, vec3 n, vec3 rd, float materialID, vec3 lightColor) {
                    if(materialID < 1.5) {
                        // Crystal material - highly reflective with dispersion
                        vec3 baseColor = mix(u_primaryColor, vec3(1.0, 0.95, 0.9), 0.7);
                        float fresnel = pow(1.0 - max(0.0, dot(-rd, n)), 3.0);
                        vec3 reflection = reflect(rd, n);
                        vec3 envColor = hsv2rgb(vec3(dot(reflection, vec3(1.0, 0.5, 0.2)) * 0.5 + 0.5, 0.6, 1.0));
                        
                        // Chromatic dispersion effect
                        float dispersion = 0.02;
                        vec3 chromatic = vec3(
                            dot(reflection, vec3(1.0, 0.0, 0.0)) * dispersion,
                            dot(reflection, vec3(0.0, 1.0, 0.0)) * dispersion,
                            dot(reflection, vec3(0.0, 0.0, 1.0)) * dispersion
                        );
                        
                        return mix(baseColor * lightColor, envColor + chromatic, fresnel * 0.9);
                        
                    } else if(materialID < 2.5) {
                        // Metallic material - high specular with colored reflections
                        vec3 baseColor = mix(u_secondaryColor, vec3(0.9, 0.9, 0.95), 0.4);
                        float metallic = u_metallicness;
                        float roughness = 0.1 + (1.0 - u_metallicness) * 0.3;
                        
                        vec3 reflection = reflect(rd, n);
                        float specular = pow(max(0.0, dot(reflection, normalize(vec3(1.0, 1.0, 1.0)))), 50.0);
                        
                        // Anisotropic highlight
                        vec3 tangent = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
                        float aniso = sin(dot(tangent, reflection) * 20.0) * 0.1 + 1.0;
                        
                        return baseColor * lightColor * (1.0 - metallic) + 
                               lightColor * specular * metallic * aniso +
                               hue2rgb(dot(reflection, vec3(0.3, 0.6, 0.1)) * 2.0) * metallic * 0.3;
                        
                    } else {
                        // Energy material - emissive with animated glow
                        float energyPhase = u_time * u_animationSpeed * 2.0 + dot(p, vec3(1.0));
                        vec3 energyColor = hsv2rgb(vec3(sin(energyPhase) * 0.5 + 0.5, 1.0, 1.0));
                        float glow = 1.5 + sin(energyPhase * 3.0) * 0.5;
                        
                        // Pulsing emission
                        float pulse = sin(u_time * u_animationSpeed * 4.0) * 0.3 + 0.7;
                        
                        return mix(u_primaryColor, energyColor, 0.8) * lightColor * glow * pulse;
                    }
                }
                
                // Advanced raymarching with adaptive stepping and early exit optimization
                void main() {
                    vec2 uv = (vUv - 0.5) * 2.0;
                    
                    // Dynamic camera system
                    vec3 ro = u_cameraPosition;
                    vec3 forward = normalize(u_cameraTarget - u_cameraPosition);
                    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
                    vec3 up = cross(right, forward);
                    vec3 rd = normalize(forward + uv.x * right * 0.6 + uv.y * up * 0.6);
                    
                    // Optimized raymarching with adaptive steps and reduced iterations
                    float t = 0.0;
                    float minDist = 1000.0;
                    vec2 sceneData;
                    float lastDist = 1000.0;
                    
                    for(int i = 0; i < 100; i++) { // Reduced from 150
                        vec3 p = ro + rd * t;
                        sceneData = map(p);
                        float d = sceneData.x;
                        minDist = min(minDist, d);
                        
                        // Early exit optimization
                        if(d < 0.002) break; // Slightly relaxed precision for performance
                        
                        // Adaptive step size - larger steps when far from surfaces
                        float stepMultiplier = d > 0.1 ? 0.9 : 0.7;
                        t += d * stepMultiplier;
                        
                        if(t > 20.0) break; // Reduced max distance
                        
                        // Prevent infinite loops
                        if(d > lastDist * 1.5) break;
                        lastDist = d;
                    }
                    
                    vec3 color = vec3(0.0);
                    
                    if(t < 19.0) { // Updated threshold
                        vec3 p = ro + rd * t;
                        vec3 n = getNormal(p);
                        float materialID = sceneData.y;
                        
                        // Cached light directions
                        vec3 light1Dir = normalize(vec3(1.0, 1.0, 1.0));
                        vec3 light2Dir = normalize(vec3(-0.7, 0.8, -0.5));
                        vec3 light3Dir = normalize(vec3(0.3, -0.6, 0.8));
                        
                        vec3 light1Color = vec3(1.0, 0.9, 0.8) * 2.0;
                        vec3 light2Color = vec3(0.4, 0.7, 1.0) * 1.5;
                        vec3 light3Color = vec3(1.0, 0.5, 0.8) * 1.2;
                        
                        // Optimized shadow calculation with reduced samples
                        float shadow1 = softShadow(p + n * 0.02, light1Dir, 0.03, 2.5, 12.0);
                        float shadow2 = softShadow(p + n * 0.02, light2Dir, 0.03, 2.5, 12.0);
                        float shadow3 = softShadow(p + n * 0.02, light3Dir, 0.03, 2.5, 12.0);
                        
                        float diff1 = max(0.0, dot(n, light1Dir)) * shadow1;
                        float diff2 = max(0.0, dot(n, light2Dir)) * shadow2;
                        float diff3 = max(0.0, dot(n, light3Dir)) * shadow3;
                        
                        // Simplified ambient occlusion with fewer samples
                        float ao = ambientOcclusion(p, n);
                        
                        // Material colors
                        vec3 col1 = getMaterialColor(p, n, rd, materialID, light1Color) * diff1;
                        vec3 col2 = getMaterialColor(p, n, rd, materialID, light2Color) * diff2;
                        vec3 col3 = getMaterialColor(p, n, rd, materialID, light3Color) * diff3;
                        
                        color = (col1 + col2 + col3) * ao;
                        
                        // Global illumination approximation
                        vec3 indirectLight = mix(u_primaryColor, u_secondaryColor, 0.5) * 0.2 * ao;
                        color += indirectLight;
                        
                        // Bloom effect for bright materials
                        if(materialID > 2.5) { // Energy material
                            float brightness = dot(color, vec3(0.299, 0.587, 0.114));
                            color += brightness * u_bloomIntensity * 0.5 * vec3(1.0, 0.8, 0.6);
                        }
                        
                        // Apply global bloom to all bright areas
                        float globalBrightness = dot(color, vec3(0.299, 0.587, 0.114));
                        if(globalBrightness > 0.6) {
                            color += (globalBrightness - 0.6) * u_bloomIntensity * vec3(1.0, 0.9, 0.7);
                        }
                        
                        // Atmospheric perspective
                        float fog = 1.0 - exp(-t * 0.02);
                        vec3 fogColor = mix(vec3(0.1, 0.15, 0.3), vec3(0.3, 0.2, 0.4), sin(u_time * u_animationSpeed * 0.1) * 0.5 + 0.5);
                        color = mix(color, fogColor, fog * 0.4);
                        
                    } else {
                        // Enhanced background with nebula effect
                        float glow = 1.0 / (1.0 + minDist * minDist * 100.0);
                        vec3 glowColor = mix(u_primaryColor, u_secondaryColor, sin(u_time * u_animationSpeed * 0.3) * 0.5 + 0.5);
                        color = glowColor * glow * 0.3;
                        
                        // Animated background nebula
                        float nebulaPhase = u_time * u_animationSpeed * 0.1;
                        vec3 nebulaColor = hsv2rgb(vec3(nebulaPhase + dot(rd, vec3(1.0)) * 0.1, 0.6, 0.3));
                        color += nebulaColor * 0.1;
                    }
                    
                    // Enhanced tone mapping and post-processing
                    color = color / (color + vec3(1.0)); // Reinhard tone mapping
                    color = mix(color, color * color * (3.0 - 2.0 * color), 0.3); // Contrast enhancement
                    color = pow(color, vec3(1.0/2.2)); // Gamma correction
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `;
            
        case 'fractal':
            return baseUniforms + `
                // Advanced material and lighting functions
                vec3 hue2rgb(float h) {
                    h = fract(h);
                    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
                }
                
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }
                
                // Fresnel reflection calculation
                float fresnel(vec3 I, vec3 N, float ior) {
                    float cosi = clamp(dot(I, N), -1.0, 1.0);
                    float etai = 1.0, etat = ior;
                    if (cosi > 0.0) { 
                        float temp = etai;
                        etai = etat;
                        etat = temp;
                    }
                    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
                    if (sint >= 1.0) return 1.0;
                    else {
                        float cost = sqrt(max(0.0, 1.0 - sint * sint));
                        cosi = abs(cosi);
                        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
                        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
                        return (Rs * Rs + Rp * Rp) / 2.0;
                    }
                }
                
                // Multiple fractal equations
                vec4 mandelbulb(vec3 p) {
                    vec3 z = p;
                    float dr = 1.0;
                    float r = 0.0;
                    float basePower = 8.0 + 4.0 * sin(u_time * u_animationSpeed * 0.3) + cos(u_seed * 0.1) * 2.0;
                    float evolutionMod = sin(u_evolutionPhase) * 0.5; // Subtle evolution
                    float power = basePower + evolutionMod;
                    int iterations = 0;
                    
                    // Optimized fractal iteration with reasonable iteration count
                    for(int i = 0; i < 12; i++) { // Reasonable iteration count for fractals
                        iterations = i;
                        r = length(z);
                        if(r > 2.5) break; // Slightly higher threshold for early exit
                        
                        float theta = acos(clamp(z.z / r, -1.0, 1.0));
                        float phi = atan(z.y, z.x);
                        dr = pow(r, power - 1.0) * power * dr + 1.0;
                        
                        float zr = pow(r, power);
                        theta = theta * power;
                        phi = phi * power;
                        
                        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
                        z += p;
                    }
                    return vec4(0.5 * log(r) * r / dr, float(iterations), r, power);
                }
                
                vec4 mandelbox(vec3 p) {
                    vec3 z = p;
                    float dr = 1.0;
                    float baseScale = 2.0 + sin(u_time * u_animationSpeed * 0.2) * 0.5;
                    float evolutionMod = cos(u_evolutionPhase * 0.7) * 0.3; // Evolution effect
                    float scale = baseScale + evolutionMod;
                    int iterations = 0;
                    
                    for(int i = 0; i < 12; i++) {
                        iterations = i;
                        // Box folding
                        if(z.x > 1.0) z.x = 2.0 - z.x;
                        else if(z.x < -1.0) z.x = -2.0 - z.x;
                        if(z.y > 1.0) z.y = 2.0 - z.y;
                        else if(z.y < -1.0) z.y = -2.0 - z.y;
                        if(z.z > 1.0) z.z = 2.0 - z.z;
                        else if(z.z < -1.0) z.z = -2.0 - z.z;
                        
                        // Sphere folding
                        float r2 = dot(z, z);
                        if(r2 < 0.25) {
                            z *= 4.0;
                            dr *= 4.0;
                        } else if(r2 < 1.0) {
                            z /= r2;
                            dr /= r2;
                        }
                        
                        z = scale * z + p;
                        dr = dr * abs(scale) + 1.0;
                        
                        if(length(z) > 4.0) break;
                    }
                    return vec4(length(z) / abs(dr), float(iterations), length(z), scale);
                }
                
                vec4 julia3D(vec3 p) {
                    vec3 z = p;
                    vec3 c = vec3(0.3 + 0.2 * sin(u_time * u_animationSpeed * 0.1), 
                                 0.5 + 0.3 * cos(u_time * u_animationSpeed * 0.15), 
                                 0.2 + 0.1 * sin(u_time * u_animationSpeed * 0.2)) + vec3(sin(u_seed * 0.1), cos(u_seed * 0.15), sin(u_seed * 0.2)) * 0.3;
                    float dr = 1.0;
                    int iterations = 0;
                    
                    for(int i = 0; i < 12; i++) {
                        iterations = i;
                        float r = length(z);
                        if(r > 2.0) break;
                        
                        // Julia iteration in 3D
                        dr = 2.0 * r * dr + 1.0;
                        float theta = acos(z.z / r);
                        float phi = atan(z.y, z.x);
                        
                        z = pow(r, 2.0) * vec3(sin(2.0 * theta) * cos(2.0 * phi), 
                                              sin(2.0 * phi) * sin(2.0 * theta), 
                                              cos(2.0 * theta)) + c;
                    }
                    return vec4(length(z) / dr, float(iterations), length(z), 2.0);
                }
                
                vec4 burningShip3D(vec3 p) {
                    vec3 z = p;
                    float dr = 1.0;
                    int iterations = 0;
                    
                    for(int i = 0; i < 12; i++) {
                        iterations = i;
                        float r = length(z);
                        if(r > 2.0) break;
                        
                        dr = 2.0 * r * dr + 1.0;
                        z = abs(z); // Burning ship modification
                        float theta = acos(z.z / r);
                        float phi = atan(z.y, z.x);
                        
                        z = pow(r, 2.0) * vec3(sin(2.0 * theta) * cos(2.0 * phi), 
                                              sin(2.0 * phi) * sin(2.0 * theta), 
                                              cos(2.0 * theta)) + p;
                    }
                    return vec4(length(z) / dr, float(iterations), length(z), 2.0);
                }
                
                vec4 tricorn3D(vec3 p) {
                    vec3 z = p;
                    float dr = 1.0;
                    int iterations = 0;
                    
                    for(int i = 0; i < 12; i++) {
                        iterations = i;
                        float r = length(z);
                        if(r > 2.0) break;
                        
                        dr = 2.0 * r * dr + 1.0;
                        z = vec3(z.x, -z.y, z.z); // Tricorn conjugate
                        float theta = acos(z.z / r);
                        float phi = atan(z.y, z.x);
                        
                        z = pow(r, 2.0) * vec3(sin(2.0 * theta) * cos(2.0 * phi), 
                                              sin(2.0 * phi) * sin(2.0 * theta), 
                                              cos(2.0 * theta)) + p;
                    }
                    return vec4(length(z) / dr, float(iterations), length(z), 2.0);
                }
                
                vec4 fractalSDF(vec3 p) {
                    float rotAngle = u_seed * 0.1 + u_time * u_animationSpeed * 0.1;
                    p.xy = mat2(cos(rotAngle), -sin(rotAngle), sin(rotAngle), cos(rotAngle)) * p.xy;
                    
                    if(u_fractalType < 0.5) return mandelbulb(p);
                    else if(u_fractalType < 1.5) return mandelbox(p);
                    else if(u_fractalType < 2.5) return julia3D(p);
                    else if(u_fractalType < 3.5) return burningShip3D(p);
                    else return tricorn3D(p);
                }
                
                // Enhanced lighting with multiple bounces
                vec3 calculateAdvancedLighting(vec3 p, vec3 n, vec3 rd, vec4 fractalData) {
                    // Multiple light sources with different colors
                    vec3 light1Dir = normalize(vec3(1.0, 1.0, 1.0));
                    vec3 light2Dir = normalize(vec3(-0.5, 0.8, -0.3));
                    vec3 light3Dir = normalize(vec3(0.0, -1.0, 0.5));
                    
                    vec3 light1Color = vec3(1.0, 0.9, 0.8);
                    vec3 light2Color = vec3(0.6, 0.8, 1.0);
                    vec3 light3Color = vec3(1.0, 0.7, 0.9);
                    
                    // Diffuse lighting
                    float diff1 = max(0.0, dot(n, light1Dir));
                    float diff2 = max(0.0, dot(n, light2Dir));
                    float diff3 = max(0.0, dot(n, light3Dir));
                    
                    vec3 diffuse = diff1 * light1Color + diff2 * light2Color * 0.6 + diff3 * light3Color * 0.4;
                    
                    // Specular highlights with varying roughness
                    float roughness = 0.1 + 0.3 * sin(fractalData.y * 0.5);
                    vec3 viewDir = -rd;
                    vec3 reflectDir1 = reflect(-light1Dir, n);
                    vec3 reflectDir2 = reflect(-light2Dir, n);
                    
                    float spec1 = pow(max(0.0, dot(viewDir, reflectDir1)), 32.0 / (roughness + 0.1));
                    float spec2 = pow(max(0.0, dot(viewDir, reflectDir2)), 16.0 / (roughness + 0.1));
                    
                    vec3 specular = spec1 * light1Color * 0.8 + spec2 * light2Color * 0.4;
                    
                    // Fresnel reflections
                    float fresnelEffect = fresnel(rd, n, 1.5);
                    vec3 envReflection = vec3(0.2, 0.4, 0.8) * fresnelEffect;
                    
                    // Subsurface scattering approximation
                    float subsurface = pow(max(0.0, dot(-light1Dir, rd)), 2.0) * 0.3;
                    vec3 sssColor = vec3(1.0, 0.4, 0.2) * subsurface;
                    
                    // Rim lighting
                    float rim = 1.0 - max(0.0, dot(viewDir, n));
                    rim = pow(rim, 3.0);
                    vec3 rimColor = vec3(0.5, 0.8, 1.0) * rim * 0.5;
                    
                    return diffuse + specular + envReflection + sssColor + rimColor;
                }
                
                void main() {
                    vec2 uv = (vUv - 0.5) * 2.0;
                    
                    // Dynamic camera system
                    vec3 ro = u_cameraPosition;
                    vec3 forward = normalize(u_cameraTarget - u_cameraPosition);
                    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
                    vec3 up = cross(right, forward);
                    vec3 rd = normalize(forward + uv.x * right * 0.6 + uv.y * up * 0.6);
                    
                    // Optimized raymarching with adaptive steps
                    float t = 0.0;
                    float minDist = 1000.0;
                    vec4 fractalInfo = vec4(0.0);
                    bool hit = false;
                    
                    for(int i = 0; i < 80; i++) { // Reduced from 128
                        vec3 p = ro + rd * t;
                        fractalInfo = fractalSDF(p);
                        float d = fractalInfo.x;
                        minDist = min(minDist, d);
                        
                        if(d < 0.002) { // Slightly relaxed precision
                            hit = true;
                            break;
                        }
                        
                        // Adaptive step size
                        float stepSize = d > 0.1 ? d * 0.8 : d * 0.6;
                        t += stepSize;
                        if(t > 12.0) break; // Reduced max distance
                    }
                    
                    vec3 color = vec3(0.0);
                    
                    if(hit && t < 11.0) { // Updated threshold
                        vec3 p = ro + rd * t;
                        
                        // Calculate normal with optimized precision
                        float eps = 0.002; // Slightly larger epsilon for performance
                        vec3 n = normalize(vec3(
                            fractalSDF(p + vec3(eps, 0, 0)).x - fractalSDF(p - vec3(eps, 0, 0)).x,
                            fractalSDF(p + vec3(0, eps, 0)).x - fractalSDF(p - vec3(0, eps, 0)).x,
                            fractalSDF(p + vec3(0, 0, eps)).x - fractalSDF(p - vec3(0, 0, eps)).x
                        ));
                        
                        // Optimized lighting
                        vec3 lighting = calculateAdvancedLighting(p, n, rd, fractalInfo);
                        
                        // Dynamic color palette based on iteration count and distance
                        float colorPhase = fractalInfo.y * 0.1 + u_time * u_animationSpeed * 0.5;
                        vec3 fractalColor1 = hsv2rgb(vec3(colorPhase, 0.8, 1.0));
                        vec3 fractalColor2 = hsv2rgb(vec3(colorPhase + 0.3, 0.9, 0.8));
                        
                        // Mix colors based on primary/secondary
                        vec3 baseColor = mix(u_primaryColor * fractalColor1, u_secondaryColor * fractalColor2, 
                                           sin(fractalInfo.y * 0.3) * 0.5 + 0.5);
                        
                        // Apply lighting
                        color = baseColor * lighting;
                        
                        // Add metallic/iridescent effects
                        float metallic = u_metallicness * (0.5 + 0.5 * sin(fractalInfo.y * 0.2));
                        vec3 iridescence = hue2rgb(fractalInfo.y * 0.05 + dot(n, rd) * 2.0) * metallic * 0.4;
                        color += iridescence;
                        
                        // Bloom effect for bright areas
                        float brightness = dot(color, vec3(0.299, 0.587, 0.114));
                        if(brightness > 0.8) {
                            color += (brightness - 0.8) * u_bloomIntensity * vec3(1.0, 0.8, 0.6);
                        }
                        
                        // Atmospheric perspective
                        float fog = 1.0 - exp(-t * 0.05);
                        vec3 fogColor = vec3(0.1, 0.2, 0.4);
                        color = mix(color, fogColor, fog * 0.3);
                        
                    } else {
                        // Enhanced background - ensure no geometry artifacts
                        color = vec3(0.0); // Start with black background
                        
                        // Subtle glow based on minimum distance to fractal
                        if(minDist < 5.0) {
                            float glow = 1.0 / (1.0 + minDist * minDist * 10.0);
                            vec3 glowColor = mix(u_primaryColor, u_secondaryColor, 0.5);
                            color += glowColor * glow * 0.1;
                        }
                        
                        // Enhanced starfield background
                        vec2 starUV = rd.xy * 15.0 + rd.z * 5.0; // 3D variation
                        vec2 starId = floor(starUV);
                        float starHash = fract(sin(dot(starId, vec2(127.1, 311.7))) * 43758.5453);
                        if(starHash > 0.996) {
                            float starBrightness = (starHash - 0.996) * 250.0;
                            vec3 starColor = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), 
                                               fract(sin(dot(starId, vec2(41.23, 67.89))) * 43758.5453));
                            color += starColor * starBrightness * 0.3;
                        }
                        
                        // Subtle nebula background
                        vec3 nebulaUV = rd * 3.0 + vec3(u_time * u_animationSpeed * 0.01);
                        float nebula = 0.0;
                        for(int j = 0; j < 3; j++) {
                            float scale = 1.0 + float(j) * 0.5;
                            nebula += sin(dot(nebulaUV * scale, vec3(1.0, 1.3, 0.7))) * 0.5 + 0.5;
                        }
                        nebula /= 3.0;
                        
                        vec3 nebulaColor = hsv2rgb(vec3(nebula * 0.2 + u_time * u_animationSpeed * 0.02, 0.4, 0.1));
                        color += nebulaColor * max(0.0, nebula - 0.3);
                    }
                    
                    // Tone mapping and gamma correction
                    color = color / (color + vec3(1.0)); // Reinhard tone mapping
                    color = pow(color, vec3(1.0/2.2)); // Gamma correction
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `;
            
        case 'noise':
            return baseUniforms + `
                // Enhanced color and utility functions
                vec3 hue2rgb(float h) {
                    h = fract(h);
                    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
                }
                
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }
                
                vec3 hash3(vec3 p) {
                    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                             dot(p, vec3(269.5, 183.3, 246.1)),
                             dot(p, vec3(113.5, 271.9, 124.6)));
                    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
                }
                
                float noise(vec3 p) {
                    vec3 i = floor(p);
                    vec3 f = fract(p);
                    vec3 u = f * f * (3.0 - 2.0 * f);
                    
                    return mix(mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                                       dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                                       dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
                               mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                                       dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                                       dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
                }
                
                // Multi-octave noise with curl noise for turbulence
                vec3 curlNoise(vec3 p) {
                    float e = 0.1;
                    float n1 = noise(p + vec3(e, 0, 0));
                    float n2 = noise(p - vec3(e, 0, 0));
                    float n3 = noise(p + vec3(0, e, 0));
                    float n4 = noise(p - vec3(0, e, 0));
                    float n5 = noise(p + vec3(0, 0, e));
                    float n6 = noise(p - vec3(0, 0, e));
                    
                    float x = n4 - n3;
                    float y = n5 - n6;
                    float z = n2 - n1;
                    
                    return normalize(vec3(x, y, z));
                }
                
                float fbm(vec3 p) {
                    float value = 0.0;
                    float amplitude = 0.5;
                    float frequency = 1.0;
                    for(int i = 0; i < 6; i++) {
                        value += amplitude * noise(p * frequency);
                        frequency *= 2.0;
                        amplitude *= 0.5;
                    }
                    return value;
                }
                
                // Enhanced volumetric rendering
                vec4 volumetricPass(vec3 ro, vec3 rd, float maxDist) {
                    vec3 color = vec3(0.0);
                    float alpha = 0.0;
                    float t = 0.1;
                    
                    // Dynamic sample count for quality
                    int samples = 50 + int(mod(u_seed * 0.1, 15.0));
                    float stepSize = maxDist / float(samples);
                    
                    for(int i = 0; i < 65; i++) {
                        if(i >= samples || t > maxDist || alpha > 0.99) break;
                        
                        vec3 p = ro + rd * t;
                        
                        // Multi-layered noise with different characteristics
                        vec3 seedOffset = vec3(
                            sin(u_seed * 0.05) * 3.0,
                            cos(u_seed * 0.07) * 3.0,
                            sin(u_seed * 0.06) * 3.0
                        );
                        p += seedOffset;
                        
                        // Animated rotation
                        float rotTime = u_time * u_animationSpeed * 0.2;
                        float rotAngle = u_seed * 0.15 + rotTime;
                        p.xz = mat2(cos(rotAngle), -sin(rotAngle), sin(rotAngle), cos(rotAngle)) * p.xz;
                        p.xy = mat2(cos(rotTime * 0.7), -sin(rotTime * 0.7), sin(rotTime * 0.7), cos(rotTime * 0.7)) * p.xy;
                        
                        // Multiple noise scales for different effects
                        float noiseScale1 = 1.5 + sin(u_seed * 0.12) * 0.8;
                        float noiseScale2 = 3.0 + cos(u_seed * 0.18) * 1.5;
                        float noiseScale3 = 6.0 + sin(u_seed * 0.24) * 2.0;
                        
                        // Layered density calculation
                        float density1 = fbm(p * noiseScale1 + vec3(0.0, u_time * u_animationSpeed * 0.1, 0.0));
                        float density2 = fbm(p * noiseScale2 + vec3(u_time * u_animationSpeed * 0.05, 0.0, 0.0));
                        float density3 = noise(p * noiseScale3 + vec3(0.0, 0.0, u_time * u_animationSpeed * 0.15));
                        
                        // Combine densities with different blending modes
                        float combinedDensity = density1 * 0.6 + density2 * 0.3 + density3 * 0.1;
                        combinedDensity = clamp(combinedDensity + 0.1, 0.0, 1.0);
                        
                        // Energy fields and plasma effects
                        vec3 curl = curlNoise(p * 2.0 + u_time * u_animationSpeed * 0.1);
                        float energyField = dot(curl, normalize(p)) * 0.5 + 0.5;
                        combinedDensity *= (0.7 + energyField * 0.3);
                        
                        // Dynamic color based on density and position
                        float colorPhase = combinedDensity + u_time * u_animationSpeed * 0.3 + length(p) * 0.1;
                        vec3 noiseColor1 = hsv2rgb(vec3(colorPhase * 0.5, 0.8, 1.0));
                        vec3 noiseColor2 = hsv2rgb(vec3(colorPhase * 0.3 + 0.3, 0.9, 0.8));
                        
                        // Mix with user colors
                        vec3 baseColor = mix(u_primaryColor * noiseColor1, u_secondaryColor * noiseColor2, 
                                           sin(colorPhase * 2.0) * 0.5 + 0.5);
                        
                        // Add energy glow effect
                        float energyGlow = pow(combinedDensity, 2.0) * 2.0;
                        vec3 energyColor = hue2rgb(colorPhase + energyField);
                        baseColor += energyColor * energyGlow * 0.3;
                        
                        // Volumetric lighting simulation
                        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
                        float lighting = max(0.3, dot(curl, lightDir) * 0.5 + 0.5);
                        baseColor *= lighting;
                        
                        // Opacity with distance falloff
                        float opacity = combinedDensity * stepSize * 2.0;
                        opacity *= exp(-t * 0.02); // Distance falloff
                        
                        // Front-to-back blending
                        color += baseColor * opacity * (1.0 - alpha);
                        alpha += opacity * (1.0 - alpha);
                        
                        t += stepSize;
                    }
                    
                    return vec4(color, alpha);
                }
                
                void main() {
                    vec2 uv = (vUv - 0.5) * 2.0;
                    
                    // Dynamic camera system
                    vec3 ro = u_cameraPosition;
                    vec3 forward = normalize(u_cameraTarget - u_cameraPosition);
                    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
                    vec3 up = cross(right, forward);
                    vec3 rd = normalize(forward + uv.x * right * 0.6 + uv.y * up * 0.6);
                    
                    // Volumetric rendering
                    vec4 volumeResult = volumetricPass(ro, rd, 8.0);
                    vec3 color = volumeResult.rgb;
                    
                    // Enhanced background with nebula
                    vec3 nebula = vec3(0.0);
                    for(int i = 0; i < 3; i++) {
                        float scale = 0.5 + float(i) * 0.3;
                        float phase = u_time * u_animationSpeed * 0.1 + float(i) * 2.0;
                        vec3 nebulaPos = rd * scale + vec3(sin(phase), cos(phase * 0.7), sin(phase * 1.3)) * 0.2;
                        float nebulaNoise = noise(nebulaPos * 3.0 + vec3(phase * 0.1));
                        vec3 nebulaColor = hsv2rgb(vec3(phase * 0.1 + nebulaNoise * 0.2, 0.6, 0.4));
                        nebula += nebulaColor * max(0.0, nebulaNoise) * 0.15;
                    }
                    
                    // Blend volume with background
                    color = mix(nebula, color, volumeResult.a);
                    
                    // Bloom effect for bright areas
                    float brightness = dot(color, vec3(0.299, 0.587, 0.114));
                    if(brightness > 0.6) {
                        color += (brightness - 0.6) * u_bloomIntensity * vec3(1.0, 0.8, 0.6);
                    }
                    
                    // Advanced tone mapping and post-processing
                    color = color / (color + vec3(1.0)); // Reinhard tone mapping
                    color = mix(color, color * color * (3.0 - 2.0 * color), 0.4); // S-curve contrast
                    color = pow(color, vec3(1.0/2.2)); // Gamma correction
                    
                    // Subtle color grading
                    color *= vec3(1.05, 1.02, 0.98); // Warm tint
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `;
            
        case 'galaxy':
            return baseUniforms + `
                // Enhanced color and utility functions
                vec3 hue2rgb(float h) {
                    h = fract(h);
                    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
                }
                
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }
                
                // Improved noise functions
                vec3 hash3(vec3 p) {
                    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                             dot(p, vec3(269.5, 183.3, 246.1)),
                             dot(p, vec3(113.5, 271.9, 124.6)));
                    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
                }
                
                float noise(vec3 p) {
                    vec3 i = floor(p);
                    vec3 f = fract(p);
                    vec3 u = f * f * (3.0 - 2.0 * f);
                    
                    return mix(mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                                       dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                                       dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
                               mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                                       dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                                   mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                                       dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
                }
                
                // Fractional Brownian Motion for galaxy structure
                float fbm(vec3 p) {
                    float value = 0.0;
                    float amplitude = 0.5;
                    float frequency = 1.0;
                    for(int i = 0; i < 5; i++) {
                        value += amplitude * noise(p * frequency);
                        frequency *= 2.0;
                        amplitude *= 0.5;
                    }
                    return value;
                }
                
                // Galaxy spiral function
                float galaxySpiral(vec2 p, float time, float armCount, float tightness) {
                    float angle = atan(p.y, p.x);
                    float radius = length(p);
                    
                    // Multiple spiral arms
                    float spiral = 0.0;
                    for(int i = 0; i < 4; i++) {
                        if(float(i) >= armCount) break;
                        float armAngle = angle + float(i) * 6.28318 / armCount;
                        float spiralAngle = armAngle - radius * tightness + time * 0.1;
                        spiral += exp(-pow(mod(spiralAngle + 3.14159, 6.28318) - 3.14159, 2.0) * 20.0);
                    }
                    
                    return spiral;
                }
                
                // Enhanced star field generation
                float starField(vec2 p, float density, float brightness) {
                    vec2 gridId = floor(p * density);
                    vec2 gridUv = fract(p * density) - 0.5;
                    
                    float hash = fract(sin(dot(gridId, vec2(127.1, 311.7))) * 43758.5453);
                    
                    if(hash > 0.98) {
                        float starSize = (hash - 0.98) * 50.0;
                        float star = exp(-dot(gridUv, gridUv) * starSize * 100.0);
                        return star * brightness;
                    }
                    
                    return 0.0;
                }
                
                void main() {
                    vec2 uv = (vUv - 0.5) * 2.0;
                    
                    // Dynamic camera system for 3D galaxy view
                    vec3 ro = u_cameraPosition;
                    vec3 forward = normalize(u_cameraTarget - u_cameraPosition);
                    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
                    vec3 up = cross(right, forward);
                    vec3 rd = normalize(forward + uv.x * right * 0.8 + uv.y * up * 0.8);
                    
                    // Galaxy plane intersection
                    float t = -ro.y / rd.y; // Intersect with y=0 plane
                    vec3 galaxyPos = ro + rd * max(t, 0.0);
                    vec2 galaxyUV = galaxyPos.xz;
                    
                    // Scale and rotation based on seed and time
                    float rotation = u_seed * 0.1 + u_time * u_animationSpeed * 0.05;
                    galaxyUV = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * galaxyUV;
                    
                    // Galaxy parameters influenced by seed
                    float galaxyScale = 3.0 + sin(u_seed * 0.15) * 1.0;
                    galaxyUV /= galaxyScale;
                    
                    float radius = length(galaxyUV);
                    float angle = atan(galaxyUV.y, galaxyUV.x);
                    
                    // Galaxy core and disk density
                    float coreDensity = exp(-radius * radius * 4.0); // Bright center
                    float diskDensity = exp(-radius * 1.5) * (1.0 - coreDensity); // Outer disk
                    
                    // Spiral arms with seed variation
                    float armCount = 2.0 + floor(mod(u_seed * 0.2, 3.0)); // 2-4 arms
                    float tightness = 2.0 + sin(u_seed * 0.3) * 1.0;
                    float spiral = galaxySpiral(galaxyUV, u_time * u_animationSpeed, armCount, tightness);
                    
                    // Combine spiral with density
                    float galaxyBrightness = (coreDensity * 2.0 + diskDensity * spiral * 1.5) * 
                                           exp(-radius * 0.8); // Overall falloff
                    
                    // Add noise for dust lanes and structure
                    vec3 noisePos = vec3(galaxyUV * 8.0, u_time * u_animationSpeed * 0.1);
                    float dustLanes = fbm(noisePos) * 0.3;
                    galaxyBrightness *= (1.0 - dustLanes * diskDensity);
                    
                    // Color variations across the galaxy
                    float colorPhase = radius * 0.5 + angle * 0.1 + u_time * u_animationSpeed * 0.1;
                    vec3 coreColor = hsv2rgb(vec3(0.05 + sin(u_seed * 0.1) * 0.1, 0.8, 1.0)); // Yellowish core
                    vec3 armColor = hsv2rgb(vec3(0.6 + cos(u_seed * 0.15) * 0.2, 0.9, 0.8)); // Blueish arms
                    vec3 dustColor = hsv2rgb(vec3(0.1 + sin(u_seed * 0.05) * 0.05, 0.6, 0.3)); // Reddish dust
                    
                    // Mix colors based on position and structure
                    vec3 galaxyColor = mix(
                        mix(armColor, coreColor, coreDensity),
                        dustColor,
                        dustLanes * 0.5
                    );
                    
                    // Apply user colors as tinting
                    galaxyColor = mix(galaxyColor, u_primaryColor, 0.3);
                    galaxyColor = mix(galaxyColor, u_secondaryColor * armColor, spiral * 0.4);
                    
                    // Multi-scale star fields
                    float stars1 = starField(galaxyUV, 50.0, 1.0); // Large bright stars
                    float stars2 = starField(galaxyUV * 2.0, 80.0, 0.6); // Medium stars
                    float stars3 = starField(galaxyUV * 4.0, 120.0, 0.3); // Small stars
                    float totalStars = stars1 + stars2 + stars3;
                    
                    // Star colors
                    vec3 starColor = mix(vec3(1.0, 0.95, 0.9), vec3(0.9, 0.95, 1.0), 
                                        fract(sin(dot(floor(galaxyUV * 50.0), vec2(12.9898, 78.233))) * 43758.5453));
                    
                    // Combine galaxy and stars
                    vec3 color = galaxyColor * galaxyBrightness + starColor * totalStars;
                    
                    // Background nebula for depth
                    if(t < 0.0 || radius > 4.0) {
                        // Looking at background space
                        vec3 bgPos = rd * 10.0 + vec3(u_time * u_animationSpeed * 0.02);
                        float nebula = fbm(bgPos * 0.5) * 0.1;
                        vec3 nebulaColor = hsv2rgb(vec3(nebula * 0.3 + 0.7, 0.4, 0.2));
                        color += nebulaColor * max(0.0, nebula);
                        
                        // Distant stars
                        float distantStars = starField(rd.xy * 20.0, 100.0, 0.2);
                        color += vec3(1.0, 0.9, 0.8) * distantStars;
                    }
                    
                    // Apply metallicness for cosmic dust shimmer
                    float shimmer = sin(colorPhase * 10.0 + u_time * u_animationSpeed * 2.0) * 0.5 + 0.5;
                    color = mix(color, color * 1.3, shimmer * u_metallicness * galaxyBrightness * 0.3);
                    
                    // Bloom effect for bright areas
                    float brightness = dot(color, vec3(0.299, 0.587, 0.114));
                    if(brightness > 0.4) {
                        color += (brightness - 0.4) * u_bloomIntensity * vec3(1.0, 0.8, 0.6);
                    }
                    
                    // Atmospheric effects and depth
                    float depth = max(t, 0.0) * 0.1;
                    vec3 atmosphereColor = mix(vec3(0.1, 0.1, 0.2), vec3(0.2, 0.1, 0.3), 
                                             sin(u_time * u_animationSpeed * 0.1) * 0.5 + 0.5);
                    color = mix(color, atmosphereColor, min(depth, 0.3));
                    
                    // Final tone mapping and post-processing
                    color = color / (color + vec3(1.0)); // Reinhard tone mapping
                    color = mix(color, color * color * (3.0 - 2.0 * color), 0.2); // Subtle contrast
                    color = pow(color, vec3(1.0/2.2)); // Gamma correction
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `;
            
        default:
            // Fallback to raymarching if unknown simulation type
            console.warn('Unknown 3D simulation type:', simulationType, 'falling back to raymarching');
            return get3DFragmentShader('raymarching');
    }
};

function init() {
    // Get the canvas element
    const canvas = document.getElementById('shaderCanvas');

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Scene
    scene = new THREE.Scene();

    // Initialize with 2D mode
    setupCamera2D();
    createMesh();

    // Event Listeners
    window.addEventListener('resize', onWindowResize, false);
    setupUIListeners();
    setupCameraControls();
}

function setupCamera2D() {
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.1, 10);
    camera.position.z = 1;
}

function setupCamera3D() {
    // Use orthographic camera for 3D raymarched scenes to avoid perspective distortion
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.1, 1000);
    camera.position.set(0, 0, 1);
}

function createMesh() {
    // Remove existing mesh with proper cleanup
    if (mesh) {
        scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (mesh.material.uniforms) {
                // Clean up any texture uniforms if they exist
                Object.values(mesh.material.uniforms).forEach(uniform => {
                    if (uniform.value && uniform.value.dispose) {
                        uniform.value.dispose();
                    }
                });
            }
            mesh.material.dispose();
        }
    }

    // Initial control values
    const initialPrimaryColor = new THREE.Color(document.getElementById('primaryColor').value);
    const initialSecondaryColor = new THREE.Color(document.getElementById('secondaryColor').value);
    const initialAnimationSpeed = parseFloat(document.getElementById('animationSpeed').value);
    const initialSeed = Math.random() * 100.0;

    // Uniforms for the shader - reuse existing uniforms object when possible
    if (!uniforms) {
        uniforms = {
            u_time: { value: 0.0 },
            u_resolution: { value: new THREE.Vector2() },
            u_primaryColor: { value: new THREE.Vector3() },
            u_secondaryColor: { value: new THREE.Vector3() },
            u_animationSpeed: { value: initialAnimationSpeed },
            u_seed: { value: initialSeed },
            u_cameraPosition: { value: new THREE.Vector3() },
            u_cameraTarget: { value: new THREE.Vector3() },
            u_bloomIntensity: { value: parseFloat(document.getElementById('bloomIntensity').value) },
            u_metallicness: { value: parseFloat(document.getElementById('metallicness').value) },
            u_fractalType: { value: 0.0 },
            u_evolutionPhase: { value: 0.0 }
        };
    }
    
    // Update uniform values
    uniforms.u_resolution.value.set(renderer.domElement.width, renderer.domElement.height);
    uniforms.u_primaryColor.value.set(initialPrimaryColor.r, initialPrimaryColor.g, initialPrimaryColor.b);
    uniforms.u_secondaryColor.value.set(initialSecondaryColor.r, initialSecondaryColor.g, initialSecondaryColor.b);
    uniforms.u_cameraPosition.value.copy(cameraControls.position);
    uniforms.u_cameraTarget.value.copy(cameraControls.target);

    // Geometry and material based on mode
    let geometry, material;
    
    if (currentMode === '2d') {
        // For 2D mode, create a plane that matches the orthographic camera bounds
        const aspect = window.innerWidth / window.innerHeight;
        geometry = new THREE.PlaneGeometry(aspect * 2, 2);
        material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShader2D,
            fragmentShader: fragmentShader2D
        });
    } else {
        // For 3D mode, use the same approach as 2D for consistency
        const aspect = window.innerWidth / window.innerHeight;
        geometry = new THREE.PlaneGeometry(aspect * 2, 2);
        material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShader3D,
            fragmentShader: get3DFragmentShader(currentSimulation)
        });
    }

    mesh = new THREE.Mesh(geometry, material);
    mesh.position.z = 0;
    scene.add(mesh);
}

function switchMode(newMode) {
    console.log('Switching mode to:', newMode, 'current simulation:', currentSimulation);
    currentMode = newMode;
    const canvas = document.getElementById('shaderCanvas');
    
    if (currentMode === '2d') {
        setupCamera2D();
        document.getElementById('simulation-selector').style.display = 'none';
        canvas.style.cursor = 'default';
    } else {
        setupCamera3D();
        document.getElementById('simulation-selector').style.display = 'block';
        canvas.style.cursor = 'grab';
        // Reset camera position for 3D mode
        cameraControls.radius = 5;
        cameraControls.azimuth = 0;
        cameraControls.elevation = 0;
        cameraControls.target.set(0, 0, 0);
        updateCameraPosition();
    }
    
    // Directly create mesh instead of using deferred update
    createMesh();
    onWindowResize(); // Ensure proper sizing
}

function switchSimulation(newSimulation) {
    console.log('Switching simulation to:', newSimulation);
    currentSimulation = newSimulation;
    if (currentMode === '3d') {
        createMesh(); // Directly recreate mesh
    }
}

function updateColorCycling() {
    if (!isColorCycling || !uniforms) return;
    
    const time = clock.getElapsedTime() * colorCycleSpeed;
    
    // Create smooth, evolving colors using HSV with cached calculations
    const primaryHue = (time * 0.1) % 1.0;
    const secondaryHue = (time * 0.1 + 0.33) % 1.0; // Triadic harmony
    
    const saturation1 = 0.8 + 0.2 * Math.sin(time * 0.3);
    const saturation2 = 0.7 + 0.3 * Math.cos(time * 0.4);
    
    const lightness1 = 0.5 + 0.3 * Math.sin(time * 0.2);
    const lightness2 = 0.4 + 0.4 * Math.cos(time * 0.25);
    
    const primaryRgb = hslToRgb(primaryHue, saturation1, lightness1);
    const secondaryRgb = hslToRgb(secondaryHue, saturation2, lightness2);
    
    // Update uniforms
    uniforms.u_primaryColor.value.set(primaryRgb[0] / 255, primaryRgb[1] / 255, primaryRgb[2] / 255);
    uniforms.u_secondaryColor.value.set(secondaryRgb[0] / 255, secondaryRgb[1] / 255, secondaryRgb[2] / 255);
    
    // Update UI color inputs (throttled to avoid excessive DOM updates)
    if (Math.floor(time * 10) % 3 === 0) { // Update UI every ~300ms
        const primaryHex = rgbToHex(primaryRgb[0], primaryRgb[1], primaryRgb[2]);
        const secondaryHex = rgbToHex(secondaryRgb[0], secondaryRgb[1], secondaryRgb[2]);
        
        document.getElementById('primaryColor').value = primaryHex;
        document.getElementById('secondaryColor').value = secondaryHex;
    }
}

function toggleColorCycling() {
    isColorCycling = !isColorCycling;
    const button = document.getElementById('cycleColorsButton');
    button.textContent = isColorCycling ? 'Stop Color Cycle' : 'Auto-Cycle Colors';
    button.style.backgroundColor = isColorCycling ? '#ff4444' : '';
}

// Color utility functions
function hslToRgb(h, s, l) {
    let r, g, b;
    
    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function generateComplementaryColors() {
    // Generate random hue (0-360 degrees)
    const primaryHue = Math.random();
    
    // Use high saturation and medium lightness for vibrant colors
    const saturation = 0.7 + Math.random() * 0.3; // 70-100% saturation
    const lightness = 0.4 + Math.random() * 0.3;  // 40-70% lightness
    
    // Calculate complementary hue (opposite on color wheel)
    const complementaryHue = (primaryHue + 0.5) % 1;
    
    // Generate colors
    const primaryRgb = hslToRgb(primaryHue, saturation, lightness);
    const secondaryRgb = hslToRgb(complementaryHue, saturation, lightness);
    
    return {
        primary: rgbToHex(primaryRgb[0], primaryRgb[1], primaryRgb[2]),
        secondary: rgbToHex(secondaryRgb[0], secondaryRgb[1], secondaryRgb[2])
    };
}

function randomizeColors() {
    const colors = generateComplementaryColors();
    
    // Update the color inputs
    const primaryColorInput = document.getElementById('primaryColor');
    const secondaryColorInput = document.getElementById('secondaryColor');
    
    primaryColorInput.value = colors.primary;
    secondaryColorInput.value = colors.secondary;
    
    // Update the shader uniforms
    const primaryColor = new THREE.Color(colors.primary);
    const secondaryColor = new THREE.Color(colors.secondary);
    
    uniforms.u_primaryColor.value.set(primaryColor.r, primaryColor.g, primaryColor.b);
    uniforms.u_secondaryColor.value.set(secondaryColor.r, secondaryColor.g, secondaryColor.b);
}

// Optimized camera control functions with throttling
function updateCameraPosition() {
    if (currentMode !== '3d') return;
    
    const now = performance.now();
    if (now - lastCameraUpdate < CAMERA_UPDATE_THROTTLE) return;
    lastCameraUpdate = now;
    
    // Convert spherical coordinates to Cartesian
    const x = cameraControls.radius * Math.cos(cameraControls.elevation) * Math.cos(cameraControls.azimuth);
    const y = cameraControls.radius * Math.sin(cameraControls.elevation);
    const z = cameraControls.radius * Math.cos(cameraControls.elevation) * Math.sin(cameraControls.azimuth);
    
    cameraControls.position.set(x, y, z);
    cameraControls.position.add(cameraControls.target);
    
    // Update uniforms only if they exist
    if (uniforms) {
        if (uniforms.u_cameraPosition) {
            uniforms.u_cameraPosition.value.copy(cameraControls.position);
        }
        if (uniforms.u_cameraTarget) {
            uniforms.u_cameraTarget.value.copy(cameraControls.target);
        }
    }
}

function handleKeyboardInput() {
    if (currentMode !== '3d') return;
    
    const moveSpeed = 0.05;
    const rotateSpeed = 0.02;
    
    // WASD movement
    if (cameraControls.keys.w || cameraControls.keys.up) {
        cameraControls.radius = Math.max(1, cameraControls.radius - moveSpeed * 5);
    }
    if (cameraControls.keys.s || cameraControls.keys.down) {
        cameraControls.radius = Math.min(20, cameraControls.radius + moveSpeed * 5);
    }
    if (cameraControls.keys.a || cameraControls.keys.left) {
        cameraControls.azimuth -= rotateSpeed;
    }
    if (cameraControls.keys.d || cameraControls.keys.right) {
        cameraControls.azimuth += rotateSpeed;
    }
    
    updateCameraPosition();
}

function setupCameraControls() {
    if (typeof window === 'undefined') return;
    
    const canvas = document.getElementById('shaderCanvas');
    
    // Mouse controls
    canvas.addEventListener('mousedown', (event) => {
        if (currentMode !== '3d') return;
        cameraControls.mouse.isDown = true;
        cameraControls.mouse.lastX = event.clientX;
        cameraControls.mouse.lastY = event.clientY;
        canvas.style.cursor = 'grabbing';
    });
    
    canvas.addEventListener('mousemove', (event) => {
        if (currentMode !== '3d' || !cameraControls.mouse.isDown) return;
        
        const deltaX = event.clientX - cameraControls.mouse.lastX;
        const deltaY = event.clientY - cameraControls.mouse.lastY;
        
        cameraControls.azimuth += deltaX * 0.01;
        cameraControls.elevation = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, cameraControls.elevation - deltaY * 0.01));
        
        cameraControls.mouse.lastX = event.clientX;
        cameraControls.mouse.lastY = event.clientY;
        
        updateCameraPosition();
    });
    
    canvas.addEventListener('mouseup', () => {
        cameraControls.mouse.isDown = false;
        canvas.style.cursor = 'grab';
    });
    
    canvas.addEventListener('mouseleave', () => {
        cameraControls.mouse.isDown = false;
        canvas.style.cursor = 'default';
    });
    
    // Mouse wheel for zoom
    canvas.addEventListener('wheel', (event) => {
        if (currentMode !== '3d') return;
        event.preventDefault();
        
        const zoomSpeed = 0.1;
        cameraControls.radius = Math.max(1, Math.min(20, cameraControls.radius + event.deltaY * zoomSpeed * 0.01));
        updateCameraPosition();
    });
    
    // Keyboard controls
    window.addEventListener('keydown', (event) => {
        if (currentMode !== '3d') return;
        
        switch(event.code) {
            case 'KeyW': cameraControls.keys.w = true; break;
            case 'KeyA': cameraControls.keys.a = true; break;
            case 'KeyS': cameraControls.keys.s = true; break;
            case 'KeyD': cameraControls.keys.d = true; break;
            case 'ArrowUp': cameraControls.keys.up = true; break;
            case 'ArrowDown': cameraControls.keys.down = true; break;
            case 'ArrowLeft': cameraControls.keys.left = true; break;
            case 'ArrowRight': cameraControls.keys.right = true; break;
        }
    });
    
    window.addEventListener('keyup', (event) => {
        switch(event.code) {
            case 'KeyW': cameraControls.keys.w = false; break;
            case 'KeyA': cameraControls.keys.a = false; break;
            case 'KeyS': cameraControls.keys.s = false; break;
            case 'KeyD': cameraControls.keys.d = false; break;
            case 'ArrowUp': cameraControls.keys.up = false; break;
            case 'ArrowDown': cameraControls.keys.down = false; break;
            case 'ArrowLeft': cameraControls.keys.left = false; break;
            case 'ArrowRight': cameraControls.keys.right = false; break;
        }
    });
}

function onWindowResize() {
    // Mark that resize is needed instead of immediate execution
    needsResize = true;
}

function handleResize() {
    if (!needsResize) return;
    needsResize = false;
    
    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;

    // Update renderer size
    renderer.setSize(newWidth, newHeight);

    // Update camera for both modes
    if (currentMode === '2d') {
        const aspect = newWidth / newHeight;
        camera.left = -aspect;
        camera.right = aspect;
        camera.top = 1;
        camera.bottom = -1;
        camera.updateProjectionMatrix();
    } else {
        // 3D mode also uses orthographic camera
        const aspect = newWidth / newHeight;
        camera.left = -aspect;
        camera.right = aspect;
        camera.top = 1;
        camera.bottom = -1;
        camera.updateProjectionMatrix();
    }

    // Update u_resolution uniform
    if (uniforms && uniforms.u_resolution) {
        uniforms.u_resolution.value.set(renderer.domElement.width, renderer.domElement.height);
    }
    
    // Only recreate mesh if mode changed, not on simple resize
    if (needsMeshUpdate) {
        createMesh();
        needsMeshUpdate = false;
    }
}

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function setupUIListeners() {
    const menuToggle = document.getElementById('menuToggle');
    const controlPanel = document.getElementById('controlPanel');
    const renderModeSelect = document.getElementById('renderMode');
    const simulationTypeSelect = document.getElementById('simulationType');
    const fractalTypeSelect = document.getElementById('fractalType');
    const primaryColorInput = document.getElementById('primaryColor');
    const secondaryColorInput = document.getElementById('secondaryColor');
    const animationSpeedInput = document.getElementById('animationSpeed');
    const speedValueDisplay = document.getElementById('speedValue');
    const bloomIntensityInput = document.getElementById('bloomIntensity');
    const bloomValueDisplay = document.getElementById('bloomValue');
    const metallicnessInput = document.getElementById('metallicness');
    const metallicValueDisplay = document.getElementById('metallicValue');
    const randomizeButton = document.getElementById('randomizeButton');
    const randomizeColorsButton = document.getElementById('randomizeColorsButton');
    const cycleColorsButton = document.getElementById('cycleColorsButton');

    menuToggle.addEventListener('click', () => {
        controlPanel.classList.toggle('open');
    });

    renderModeSelect.addEventListener('change', (event) => {
        switchMode(event.target.value);
    });

    simulationTypeSelect.addEventListener('change', (event) => {
        switchSimulation(event.target.value);
        // Show/hide fractal options
        const fractalOptions = document.getElementById('fractal-options');
        fractalOptions.style.display = event.target.value === 'fractal' ? 'block' : 'none';
    });

    fractalTypeSelect.addEventListener('change', (event) => {
        if (!uniforms) return;
        const fractalTypeMap = {
            'mandelbulb': 0.0,
            'mandelbox': 1.0,
            'julia': 2.0,
            'burning_ship': 3.0,
            'tricorn': 4.0
        };
        currentFractalType = event.target.value;
        uniforms.u_fractalType.value = fractalTypeMap[event.target.value];
    });

    primaryColorInput.addEventListener('input', debounce((event) => {
        if (!uniforms) return;
        const color = new THREE.Color(event.target.value);
        uniforms.u_primaryColor.value.set(color.r, color.g, color.b);
    }, 50));

    secondaryColorInput.addEventListener('input', debounce((event) => {
        if (!uniforms) return;
        const color = new THREE.Color(event.target.value);
        uniforms.u_secondaryColor.value.set(color.r, color.g, color.b);
    }, 50));

    animationSpeedInput.addEventListener('input', debounce((event) => {
        if (!uniforms) return;
        const speed = parseFloat(event.target.value);
        uniforms.u_animationSpeed.value = speed;
        speedValueDisplay.textContent = speed.toFixed(1);
    }, 50));

    bloomIntensityInput.addEventListener('input', debounce((event) => {
        if (!uniforms) return;
        const bloom = parseFloat(event.target.value);
        uniforms.u_bloomIntensity.value = bloom;
        bloomValueDisplay.textContent = bloom.toFixed(1);
    }, 50));

    metallicnessInput.addEventListener('input', debounce((event) => {
        if (!uniforms) return;
        const metallic = parseFloat(event.target.value);
        uniforms.u_metallicness.value = metallic;
        metallicValueDisplay.textContent = metallic.toFixed(2);
    }, 50));

    randomizeButton.addEventListener('click', () => {
        if (!uniforms) return;
        uniforms.u_seed.value = Math.random() * 100.0;
    });

    randomizeColorsButton.addEventListener('click', () => {
        randomizeColors();
    });

    cycleColorsButton.addEventListener('click', () => {
        toggleColorCycling();
    });
}

function animate() {
    requestAnimationFrame(animate);

    // Handle resize if needed
    handleResize();

    // Handle camera controls for 3D mode (throttled)
    if (currentMode === '3d') {
        handleKeyboardInput();
    }

    // Update color cycling
    updateColorCycling();

    // Update uniforms efficiently
    if (uniforms) {
        const elapsedTime = clock.getElapsedTime();
        uniforms.u_time.value = elapsedTime;
        uniforms.u_evolutionPhase.value = elapsedTime * 0.1; // Slow evolution
        
        // Only update camera uniforms if they exist and we're in 3D mode
        if (currentMode === '3d') {
            if (uniforms.u_cameraPosition) {
                uniforms.u_cameraPosition.value.copy(cameraControls.position);
            }
            if (uniforms.u_cameraTarget) {
                uniforms.u_cameraTarget.value.copy(cameraControls.target);
            }
        }
    }

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