//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ursuleac Zsolt
// Neptun : S8H56Y
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"


//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
	//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;
struct Material;
class Paramsurface;
class Rect;
struct Object;

Material* mat1 ;
Rect* rekt ;
Object* rektO ;
const int tessellationLevel = 500;
const vec3 LOOKAT = vec3(2,1,1);//vec3(2.5f, 2.5f, 2.5f);
bool released = false;
//---------------------------
struct Camera { // 3D camera
	//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth/2 / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 50;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};
class PhongShader2 : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
		out float height;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
			height = vtxPos.z;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		in float height;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			//vec3 texColor = (height > 1.5) ? vec3(0.5, 0.25f, 0) : (height < 0) ? vec3(0.2, 1, 0) : (height * vec3(0.5, 0.25f, 0) + (1.5-height) * vec3(0.2, 1, 0) )/1.5;//texture(diffuseTexture, texcoord).rgb;
			//min -1.093894  max: 1.063786			
			float intensity = (height + 1.293894)/(1.063786+1.093894);
			vec3 texColor = intensity*vec3(0.5, 0.25f, 0) + (1-intensity)*vec3(0.2, 1, 0);
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader2() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};
class PhongShader3 : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
 
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer
 
		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader3() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
protected:
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {};

	virtual VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	virtual void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};
const float SCALE = .1f;
const float m = 1;
const float D = 3.0f;
const float l0 = 2.0f;
const float ro = .05f;
const float kappa = .03f;
const float a = 2 * SCALE;
const float b = 5 * SCALE;
const float c = 2 * SCALE;
const float I = m * (a * a + c * c) / 12;
const vec3 g = vec3(0, 0, -5.0f);
const vec3 sV = vec3(2.0f,0,6);
class Rect : public ParamSurface {
public:
	vec3 p1, p2, p3, p4, p5, p6, p7, p8;
	vec3 pCenter;
	const int K = 6;

	vec3 p = (0, 0, 0); //momentum
	vec3 L = (0, 0, 0); //angular momentum


	std::vector<vec3> f[8] = {};
	vec3 offset = vec3(0,0,0);
	Rect() {
		p1 = vec3(2.5f, 1, -1)+ offset;
		p2 = vec3(-2.5f, 1, -1)+ offset;		//p1, p2, p6, p5 left
		p3 = vec3(-2.5f, -1, -1)+ offset;		//p3, p4, p8, p7 right
		p4 = vec3(2.5f, -1, -1)+ offset;		//p2, p3, p7, p6 back
		p5 = vec3(2.5f, 1, 1)+ offset;
		p6 = vec3(-2.5f, 1, 1)+ offset;
		p7 = vec3(-2.5f, -1, 1)+ offset;
		p8 = vec3(2.5f, -1, 1)+ offset;

		for (int i = 0; i < 6; i++)
			f[i] = std::vector<vec3>();
		

		//lower side
		f[0].push_back(p4);
		f[0].push_back(p3);
		f[0].push_back(p2);
		f[0].push_back(p1);
		f[0].push_back(vec3(0, 0, -1));

		//upper side
		f[1].push_back(p5);
		f[1].push_back(p6);
		f[1].push_back(p7);
		f[1].push_back(p8);
		f[1].push_back(vec3(0, 0, 1));

		//nose side
		f[2].push_back(p4);		
		f[2].push_back(p1);
		f[2].push_back(p5);
		f[2].push_back(p8);
		f[2].push_back(vec3(1, 0, 0));

		//left side
		f[3].push_back(p1);
		f[3].push_back(p2);
		f[3].push_back(p6);
		f[3].push_back(p5);
		f[3].push_back(vec3(0, 1, 0));

		//right side
		f[4].push_back(p3);
		f[4].push_back(p4);
		f[4].push_back(p8);
		f[4].push_back(p7);
		f[4].push_back(vec3(0, -1, 0));

		//back side
		f[5].push_back(p2); 
		f[5].push_back(p3);
		f[5].push_back(p7);
		f[5].push_back(p6);
		f[5].push_back(vec3(-1, 0, 0));

		//for (int k = 0; k < 2; k++)
		create();
		setCenter();
	}

	VertexData GenVertexRectData(float u, float v, int k) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);

		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));

		float x=0, y=0, z=0;
		if (k == 0 || k == 1)	//lower or upper base
		{
			int firstP = (k == 0) ? 1 : 2;
			int nextP = (k == 0) ? 0 : 3;
			int upP = (k == 0) ? 2 : 1;

			x = u * length(f[k].at(nextP) - f[k].at(firstP)) + f[k].at(firstP).x;
			y = v * length(f[k].at(upP) - f[k].at(firstP)) + f[k].at(firstP).y;
			z = f[k].at(0).z;
		}
		if(k == 3 || k == 4){
			int firstP = (k == 4) ? 0 : 1;
			int nextP = (k == 4) ? 1 : 0;
			int upP = (k == 4) ? 3 : 2;

			z = v * length(f[k].at(upP) - f[k].at(firstP)) + f[k].at(firstP).z;
			y = f[k].at(firstP).y;
			x = u * length(f[k].at(nextP) - f[k].at(firstP)) + f[k].at(firstP).x;
		}
		
		if (k == 2 || k == 5) {
			int firstP = (k == 2) ? 0 : 1;
			int nextP = (k == 2) ? 1 : 0;
			int upP = (k == 2) ? 3 : 2;

			z = v * length(f[k].at(upP) - f[k].at(firstP)) + f[k].at(firstP).z;
			x = f[k].at(0).x;
			y = u * length(f[k].at(nextP) - f[k].at(firstP)) + f[k].at(firstP).y;
		}
		
		

		vtxData.position = vec3(x,y,z);
		vtxData.normal = vec3(0, 0, 1);//vec3(f[k].back());
		return vtxData;
	}

	

	virtual void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for(int k = 0; k<K; k++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j <= M; j++) {
					vtxData.push_back(GenVertexRectData((float)j / M, (float)i / N, k));
					vtxData.push_back(GenVertexRectData((float)j / M, (float)(i + 1) / N, k));
				}
		glBufferData(GL_ARRAY_BUFFER, K*nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < K * nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
	void setCenter() {
		vec3 plow = GenVertexRectData(0, 0, 3).position;
		vec3 phigh = GenVertexRectData(1,1, 4).position;
		pCenter = plow + (-plow + phigh) / 2;
	}
};

class Field : public ParamSurface {
	const int OFFS = 5;
	float xMin = OFFS, xMax = 10, yMin = OFFS, yMax = 10, A0 = 0.2;
	static const int N = 9;
	float phi[N][N] = {};
public:
	Field() {
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) 
				phi[i][j] = (float)rand() / RAND_MAX  * 2*(float)M_PI;
		create();
	}
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		U = U * xMax - xMin, V = V * yMax - yMin;
		Dnum2 H = (0, 0);
		for (int f1 = 0; f1 < N; f1++)
			for (int f2 = 0; f2 < N; f2++) {
				if (f1 + f2 <= 0)
					continue;
				H = H + Cos(U * f1 + V * f2 + phi[f1][f2]) * (A0 / sqrtf(f1 * f1 + f2 * f2));
			}

		vd.position = vec3(U.f, V.f, H.f);
		vd.normal = cross(vec3(U.d.x, V.d.x, H.d.x)/*drdU*/, vec3(U.d.y, V.d.y, H.d.y)/*drdV*/);
		
		return vd;
	}
	
};
//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis, rotationAxis2,
		rotationAxisHanging;
	float rotationAngle, rotationAngle2 = 0, rotationAngleHanging;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(3,2,1), rotationAngle(0), rotationAxis2(0,1,0), rotationAxisHanging(0,0,0), rotationAngleHanging(0){
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation)  * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle += 0.8f * tend; }
	virtual void Bigger(float dt) { scale = (1+dt) * scale; }
};

void iniObjs() {
	mat1 = new Material();
	mat1->kd = vec3(0.5, 0.5, 0);
	mat1->ks = vec3(0.1, 0.1, 0);
	mat1->ka = vec3(0.2f, 0.2f, 0.2f);
	mat1->shininess = 25;
	rekt = new Rect();
	rektO = new Object((Shader*)new PhongShader3(), mat1, new CheckerBoardTexture(15, 20), rekt);
	rektO->rotationAxis = vec3(0, 1, 0);
	rektO->rotationAngle = M_PI / 2;
}
vec3 midPoint() {
	vec3 midP = rektO->translation;
	return midP;
}

vec3 upPoint() {
	vec3 iniup = (-rekt->p1 + rekt->p5) / 2*SCALE;
	vec4 iup = vec4(iniup.x, iniup.y, iniup.z, 1) * RotationMatrix(rektO->rotationAngle, rektO->rotationAxis) * TranslateMatrix(rektO->translation);
	iup = iup / iup.w;
	return vec3(iup.x, iup.y, iup.z);
}

vec3 facePoint() {
	vec3 iniface = vec3(rekt->f[2].at(0) + (-rekt->f[2].at(0) + rekt->f[2].at(2)) / 2);
	vec4 iface = vec4(iniface.x, iniface.y, iniface.z, 1) * RotationMatrix(rektO->rotationAngle, rektO->rotationAxis) * TranslateMatrix(rektO->translation);
	iface = iface / iface.w;
	vec3 faceP = vec3(iface.x, iface.y, iface.z);
	return faceP;
}
vec3 l() {
	return midPoint() - SCALE * (-midPoint() + facePoint());
}


//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
	Camera TPScam;
	Camera FPScam;
public:
	void Build() {
		// Shaders
		Shader* phongShader2 = new PhongShader2();
		Shader* phongShader3 = new PhongShader3();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.5,0.5,0); vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.1,0.1,0); vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 25;

		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);

		Geometry* field = new Field();
		Object* fieldObject = new Object(phongShader2, material1, texture15x20, field);
		fieldObject->rotationAxis = vec3(1, 0, 0);
		objects.push_back(fieldObject);

		rektO->translation = vec3(-midPoint() + vec3(0,0,3));
		rektO->scale = vec3(SCALE, SCALE, SCALE);
		objects.push_back(rektO);
		
		

		// Camera
		TPScam.wEye = vec3(4, 4, 3.2);
		TPScam.wLookat = vec3(LOOKAT);
		TPScam.wVup = vec3(0, 0, 1);

		FPScam.wEye = facePoint();
		FPScam.wLookat = midPoint() + (facePoint() - midPoint()) * 2;
		FPScam.wVup = vec3(0, 0, 1);

		camera = TPScam;

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 9, 0);		// ideal point -> directional light source
		lights[0].La = vec3(0.5f, 0.5f, 0.5f);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-5, 5, 9, 0);	// ideal point -> directional light source
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);
	}
	void setFaceView() {
		vec3 fp = midPoint();
		FPScam.wEye = fp;
		FPScam.wLookat = (midPoint() + SCALE*(facePoint() - midPoint()) * 2);
		FPScam.wVup = -midPoint() + upPoint();
	}

	void setFPS() {
		camera = FPScam;
	}

	void setTPS() {
		camera = TPScam;
	}
	
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
		
	}

	void Animate(float tstart, float tend) {
		//for (Object* obj : objects) obj->Animate(tstart, tend);
		objects.front()->Animate(tstart, tend);
	}
	void Bigging(float dt) {
		TPScam.wEye = TPScam.wLookat+(-TPScam.wLookat + TPScam.wEye)*dt;
	}
	void RotateCam(float dt) {
		vec4 newCam4= vec4(TPScam.wEye.x, TPScam.wEye.y, TPScam.wEye.z, 1.0f)*TranslateMatrix(-TPScam.wLookat )*RotationMatrix(dt, vec3(0,0,1))*TranslateMatrix(TPScam.wLookat);
		newCam4 = newCam4 / newCam4.w;
		TPScam.wEye = vec3(newCam4.x, newCam4.y, newCam4.z);
	}
};

Scene scene;

void Step(float dt) {
	if (!released)
		return;
	
	vec3 K = D * normalize(sV - l()) * (length(sV - l()) - l0);

	vec3 F = m * g + ((length(sV - l()) > l0) ? K : vec3(0,0,0)) - ro * rekt->p / m;
	rekt->p = rekt->p + F * dt;
		
	vec3 M = cross((l() - midPoint()), K) - kappa * rekt->L/I;
	rekt->L = rekt->L + M * dt;
		
	rektO->translation = rektO->translation + rekt->p / m * dt;
	rektO->rotationAngle = rektO->rotationAngle + ((dot(rekt->L, vec3(0, 1, 0)) >= 0) ? 1 : -1) * length(rekt->L / I * dt);

}

void printInfo() {
	vec3 t = rektO->translation;
	printf("rektPulse = %lf,%lf,%lf\n\n", t.x, t.y, t.z);
}

// Initialization, create an OpenGL context
void onInitialization() {
	
	glViewport(windowWidth/4, windowHeight, windowWidth/2, windowHeight);
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	iniObjs();
	scene.Build();
	scene.Bigging(1.5f);
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	glViewport(0, 0, windowWidth/2, windowHeight);
	scene.setFPS(); 
	scene.Render();
	glViewport(windowWidth/2, 0, windowWidth / 2, windowHeight);
	scene.setTPS();
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}
bool pressed[256] = { false, };
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	pressed[key] = true;
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	pressed[key] = false;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.01f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	if (pressed['v'])
		scene.Animate(tend, dt);
	
	if (pressed['z'])
		scene.Bigging(0.8);
	if (pressed['x'])
		scene.Bigging(1.1);

	if (pressed['q'])
		scene.RotateCam(-dt);
	if (pressed['e'])
		scene.RotateCam(dt);
	
	

	//Step((float)(tend - tstart));
	if (pressed[' ']) {
		released = true;
	}

	if (pressed['i'])
		printInfo();
	
	for (float i = 0; i < tend - tstart; i += dt) {
		Step(dt);
		scene.RotateCam(dt);
	}
	scene.setFaceView();
	
	glutPostRedisplay();
}