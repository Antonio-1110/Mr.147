"""Per-pixel lighting with shadows (GLSL 1.20, runs on macOS and Linux).

Panda3D 1.10's automatic shader needs Cg, which isn't available on Apple
Silicon, so we use our own small shader instead.
"""

MAX_LIGHTS = 6

VERT = """
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;
varying vec3 v_pos;
varying vec3 v_nrm;
varying vec2 v_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_pos = vec3(p3d_ModelViewMatrix * p3d_Vertex);
    v_nrm = normalize(p3d_NormalMatrix * p3d_Normal);
    v_uv = p3d_MultiTexCoord0;
}
"""

FRAG = """
#version 120
#define N %d
uniform struct p3d_LightSourceParameters {
    vec4 color;
    vec4 position;
    vec3 spotDirection;
    float spotExponent;
    float spotCosCutoff;
    vec3 attenuation;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[N];
uniform struct { vec4 ambient; } p3d_LightModel;
uniform struct {
    vec4 ambient; vec4 diffuse; vec4 emission; vec3 specular; float shininess;
} p3d_Material;
uniform sampler2D p3d_Texture0;
// Reflections of the surroundings on shiny balls: the cloth below and the room above.
uniform mat4 trans_world_to_apiview;
uniform vec4 env_floor;   // cloth colour as it comes out in the picture
uniform vec4 env_sky;     // room colour
uniform vec4 env_gloss;   // x: how strongly this object mirrors its surroundings (0 = not at all)
varying vec3 v_pos;
varying vec3 v_nrm;
varying vec2 v_uv;
void main() {
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    vec3 v = normalize(-v_pos);
    vec3 base = p3d_Material.diffuse.rgb * texture2D(p3d_Texture0, v_uv).rgb;
    vec3 col = p3d_LightModel.ambient.rgb * base;
    for (int i = 0; i < N; ++i) {
        vec3 lc = p3d_LightSource[i].color.rgb;
        if (dot(lc, lc) < 1e-6) continue;
        vec3 ld = p3d_LightSource[i].position.xyz - v_pos * p3d_LightSource[i].position.w;
        float dist = length(ld);
        vec3 l = ld / dist;
        float spot = max(dot(-l, normalize(p3d_LightSource[i].spotDirection)), 0.0);
        float atten = pow(spot, p3d_LightSource[i].spotExponent) /
            dot(p3d_LightSource[i].attenuation, vec3(1.0, dist, dist * dist));
        vec4 sc = p3d_LightSource[i].shadowViewMatrix * vec4(v_pos, 1.0);
        float shadow = 0.0;  // 3x3 soft shadow
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                shadow += shadow2DProj(p3d_LightSource[i].shadowMap,
                    sc + vec4(dx, dy, 0.0, 0.0) * sc.w * 0.0015).r;
        shadow /= 9.0;
        float ndl = max(dot(n, l), 0.0);
        vec3 h = normalize(l + v);
        float spec = ndl > 0.0 ? pow(max(dot(n, h), 0.0), p3d_Material.shininess) : 0.0;
        col += lc * atten * shadow * (base * ndl + p3d_Material.specular * spec);
    }
    if (env_gloss.x > 0.0) {
        // Real balls mirror the cloth round their lower edge (a black 8 on blue cloth has
        // a blue rim) and the room above. Fresnel: little head-on, strong at grazing angles.
        vec3 up = normalize((trans_world_to_apiview * vec4(0.0, 0.0, 1.0, 0.0)).xyz);
        vec3 r = reflect(-v, n);
        vec3 env = mix(env_floor.rgb, env_sky.rgb, smoothstep(-0.15, 0.25, dot(r, up)));
        float fres = 0.2 + 0.8 * pow(1.0 - max(dot(n, v), 0.0), 2.0);  // stronger than ideal resin: matched to the real 8 on blue cloth (camera haze and blur add to it)
        col += env_gloss.x * fres * env;
        // Light bounced off the cloth onto the ball's underside.
        col += 0.25 * env_gloss.x * base * env_floor.rgb * max(-dot(n, up), 0.0);
    }
    gl_FragColor = vec4(col, 1.0);
}
""" % MAX_LIGHTS
