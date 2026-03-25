# The Steel Sphere Rendering Run
## Introduction
Hello! Welcome to ACM SIGGRAPH @ UNLV's bizzare inverse rendering event. We will be implementing everything with taichi-lang python so that our code is preformant and modular. The challenge has two primary parts first we will create a fairly robust path tracer before moving onto actual differential/inverse part of the competition. The second half, our differential path tracer, will require you to implement various functions that compute gradients (will explain what this means in the relevant section) and the scheduler that will run the training kernels. Nevermind that however, let the Steel Ball Run BEGIN!
## Challenge 1
In the first phase of the race competitors are faced with a 10 km stretch of desert. Unknown to them a force greater than the laws of nature is witnessed... THE POWER OF SPIN! In order to keep up competitors must breakdown and analyze what power the steel balls hold. The balls hold special properties related to their rotation... hold on? How do we even know they're rotating? Oh... well if we can look at reflections on the spheres surface and semi-random sampling of their surfaces we can determine if they're truly *spinning*. We shall begin by looking at the code in ```render_targets.py``` and fill out every function with the comment ```# Challenge 1``` then run ```uv run test.py --challenge 1```.
## Analytical Functions
### Hash_u32
This will be our function to generate what is called a seed, a seed is an initial value used to begin a pseudo-random generation of values. With these seeds we can create of range of values that are close enough to our expected values.
```
  First select an initial initial value called val
  To grow our seed we have 5 steps:
    Step 1: val = val XOR (val >> 16)
    
    What this step does:
      Lets say val is some 32 bit value
      val:  1010 1011 1100 1101 0001 0010 0011 0100
      Then shift it to the right 16 places creating temp
      temp: 0000 0000 0000 0000 1010 1011 1100 1101
      Finally we preform a xor operation like so:
      1010 1011 1100 1101 0001 0010 0011 0100 (ABCD1234)
      ^ 0000 0000 0000 0000 1010 1011 1100 1101 (0000ABCD)
      -----------------------------------------
      1010 1011 1100 1101 1011 1001 1111 1001 (ABCDB9F9)
    
    Step 2: val = val * ti.u32(0x45D9F3B)
    
    What this step does:
      Multiplies our val by 73,244,475
    
    Step 3: repeat Step 1
    
    Step 4: repeat Step 2:
    
    Step 5: repeat Step 1 and return val
```
### Next_Rand
```
This function calls our hash function and stores its results in our rng_seed field. Fairly simple implementation we simply call the hash_u32 function with the argument of rng_seed[px,py] then store that result in rng_seed[px,py], finally we should recast our rng_seed[px,py] as a taichi float 32, divide the seed by 4294967295.0 and return the result.
```
### GGX_D
```
The Trowbridge-Reitz Normal Distribution Function. 
D_GGX(h) = α² / (π · ((n·h)² · (α² - 1) + 1)²)
(n·h) is the dot product of our normal and half-vector (will make more sense in challenge 2)
(n·h) is given as ndoth
α = roughness²
if α = 0 then specular highlights are similar to a mirror otherwise α = 1 gives a diffuse-like spread.
```
### Smith_G1
```
The Smith G1 functino is a microfacet shadowing/masking term used to determine how much of a surface's microgeometry is visible from a specific viewing direction  (w_0) or light direction (w_i). Commonly used with the Trowbridge_Reitz distribution, the Smith Lambda function for GGX is given:
Λ(v) = (-1 + sqrt(1 + α²tan²θ)) / 2
This is disgusting... thankfully with algebra we can reduce it by substituting tan²θ = (1 - cos²θ) / cos²θ.
Our final equation and the function we need to implement is given as:
G1 = 2·NdotV / (NdotV + sqrt(α² + (1 - α²)·NdotV²))
```
### Fresnel_Schlick
```
Shlick's appromixation for the Fresnel factor, which determines the ratio of reflected light from a surface based on the viewing angle.
The equation is given as:
F(cos_theta, f0) = f0+(1-f0)(1-cos_theta)^5
```
## Challenge 2
Aware now of properties beyond you feel another presence that you're in command of. COOK-TORRANCE MICROFACET BRDF awakens as your stand in the Steel Ball Run, but before still the spin alludes you requiring you to become framiliar with a new piece of yourself. Your new power of a standard physically based rendering shading model that supports metallic-roughness pipelines is now your primary focus. Look at functions again in ```render_targets.py``` with the commment ```# Challenge 2``` and fill them out, once finished run ```uv run test.py --challenge 2```.
### BRDF
```
Lets look at eval_brdf in render_targets.py.

Your stands first ability is Bidirectional Reflection Distribution Function (BRDF), it describes how light reflects off a surfaced based on viewing and illumination angles. Terms of this ability are D, F, and G. D is our result from GGX_D, F is our result from Fresnel_Shlick, and G is the result from Smith_G1.

Our brdf takes the inputs:
wo = outgoing direction (toward camera)
wi = incoming direction (toward light or bounce)
n = surface normal
base_color = albedo
rough = roughness coefficient 
metal = metallic coefficient

To evaluate our brdf correctly we need to find ndotl, ndotv, h, ndoth, vdoth, alpha, and f0.
ndotl = max(dot(n, wi), 0.0)
ndotv = max(dot(n, wo), EPS)
h = normalize(wo+wi)
ndoth = max(dot(n,h), 0.0)
vdoth = max(dot(wo,h), 0.0)
alpha = max(rough^2, 0.001)
f0 = (1.0-metal)*(0.04,0.04,0.04)+metal*base_color

Now we can compute our terms D, F, and G.
Finally, we can evaluate the components of light for our model.

specular = F*(D*G/(4.0*ndotv*ndotl+EPS))
kd = (1.0-F)*(1.0-metal) # diffuse reflectance coefficient
diffuse = kd*base_color/PI

Our full evaluation is returned as:
result = (diff+spec)*ndotl
```
### Intersection
Look at scene_intersect in render_targets.py

Now things get tricky as we will now need to loop and preform intersectin testing for our rays.

Our function takes two parameters ray_o (ray_origin) and ray_d (ray_direction), these make up a single ray that will then be tested against every primitive in our scene return the distance along the ray, surface normal at the hit, what material was hit, and whether the hit surface is an emitter.

In regards to primitives in our scene we only support quads and spheres, so we will need two intersection loops that compares against all quads and all spheres respectively.

For both the sphere and quad loop we need to first compare our current index to the number of actual quads and spheres in our scene. Something like:
```
for quadId from 0 to MAX_QUADS:
  if quadId < nq:
    ...

for sphereId from 0 to MAX_SPHERES:
  if sphereId < ns:
    ...
```

#### Quad Intersection:
Each quad is defined by a center point, a normal, and two tangent vectors u and v that define its extent.
Our proposed intersection works in two steps. 

First we intersect the ray with the infinite plane containing the quad. The plane equation is ```dot(p-center,normal) = 0``` where ```p = ray_o+t*ray_d```, saving you the need to do algebra in order to solve for t we can rewrite our equation as: ```t = dot(center-ray_o,normal) / dot(ray_d,normal)```. Thing to note is if the denominator is near zero the ray is basically parallel to the plane so we should skip it entirely.

Secondly we check if the hit point is within the quad's bounds. We project the vector from the center to hit point onto each tangent axis, this looks like:
```
  pu = dot(p,u) / dot(u,u)
  pv = dot(p,v) / dot(v,v)
```
Make sure that both pu and pv are within the range of [-1,1] then save closest_t, hit_normal, hit_mat, and hit_light values.

#### Sphere Intersection:
A sphere is given as ```|p - center|^2=r^2```, substitute the same function for p as we did in the quad intersection section and we get a new quadratic equation after some algebra:
```
at^2 + bt + c = 0
However since we substituted p as function of our ray we need to change our center in regards to the ray also,
oc = ray_o - center
|oc + t * ray_d|^2 = r^2
Expanded form (what you implement):
dot(ray_d,ray_d) * t^2 + 2 * dot(oc,ray_d) * t + dot(oc,oc) - r^2 = 0
```

We then solve the equation if our discriminant is negative then there is no intersection, otherwise take the smaller root, but if its behind the ray origin then try the largest root. Compute the normal which is just the normalized vector from sphere center to hit point and save the same information as the quad intersection part. 

### Visibility
This function will be core to testing whether an intersection matters or not because it determines if the shading point can see the light or is obscured by another object. We take in two parameters p, the shading point, and target, the sampled point on the light source. To test if light reachs the shading point we shoot a shadow ray (vector) from the target to our shading point if the closest hit distance returned by our intersection is closer than the distance to the light from our shading point then something is in the way. A check for this is something like ```closest < dist - 0.001```.

If something is in the way return 0 otherwise if visible return 1


## Challenge 3
### Note: Make sure your code so far passes all the checks done by challenge 1 and 2

The power of your stand now rivals that of spin, targeting Gyro Zepelli with the tools obtained will now allow you to properly analyze spin at its most basic form... THE GROUND TRUTH OF GOLDEN SPIN!!! To beat Zepelli and get the chance to study the steel balls you must use utilize analytical functions and path trace the balls at any defined sampling amount!

Our rendering function ```render_high_spp``` found in ```render_targets.py``` takes one parameter spp or samples per pixel. This determines the amount of independent rays that are shot through each pixel and averaged together. More samples equals a less noisy image.

Our renderer will perform the Monte Carlo estimate of each pixel's radiance and each sample will be an independent estimate of the rendering equation, and averaging them converges torwards the correct image (ground truth image).

In order to render our scene to produce ground truth images we need to follow a few steps listed below.

For every pixel and for every sample we want to initialize a fresh random seed by hasing together pixel coordinates, sample index, and view index. This ensures every sample gets a unique but deterministic random sequence.

We then generate a primary ray and jitter the pixel position by a random subpixel offset (jx, jy) for anti-aliasing, convert to NDC coordinates in [-1,1], then build a ray from the camera position along a direction constructed from the camera basis vectors multiplied by some coefficient to alter field of view. The camera basis vectors should look something like:
```
ray_o = cam_pos[view]
ray_d = tm.normalize(
        cam_fwd[view]
        + cam_right[view] * u_coord * 0.6
        + cam_up_vec[view] * v_coord * 0.6
      )
```
Once our prelimary values are prepared we can begin our bounce loop for up to our determined MAX_DEPTH. While a ray is still alive it will:
1. Intersect the scene: If nothing is hit, kill the ray. If the light is hit directly on bounce 0, add the light's emission so the light source is visible in the image, then kill the ray. On later bounces, direct light hits are ignored to avoid double-counting with NEE
2. Evaluate Direct Lighting (NEE): Pick a random point on the area light by offsetting from its center along its two tangent axes with random parameters in [-1, 1]. Compute the direction and distance to that light sample. Check that the light faces the surface and the surface faces the light (ndl > 0, lc > 0). If so, test visibility with a shadow ray. If unoccluded, evaluate the BRDF, apply the geometry term (cos_light / distance²) and the light PDF (1 / area), and add the contribution scaled by current throughput.
3. Perform an indirect bounce: Build a local coordinate frame around the surface normal, then sample a cosine-weighted hemisphere direction. Evaluate the BRDF for that direction, divide by the cosine-hemisphere PDF (cos(θ) / π), and multiply into the throughput this carries the accumulated path weight forward.
4. Roussian Roulette: Compute a survival probability from the brightest throughput channel, clamped to [0.05, 0.95]. Draw a random number, if it exceeds the survival probability, kill the ray. Otherwise, boost throughput by 1/probability to keep the estimator unbiased. This lets dim paths die early without wasting compute, while bright paths survive proportionally more often. Occurs after 2 or more bounces take place.
5. Advance the ray: Set the new origin to the hit point (offset slightly along the normal to avoid self-intersection) and the new direction to the sampled hemisphere direction.

Once our bounce loop has finished we divide the accumulated color by our spp amount and write the result to our image buffer then process the next pixel.

If you feel confident that your renderer is correct or you want to test its output when you're writing to the image buffer run the command ```uv run render_targets.py --scenes/cornellbox.json --test-render```. This will open a window with your beautifully rendered scene and be the confirmation for you to move on to the next phase!

# Conclusion of Basic Rendering
## Introduction to Inverse/Differential Rendering
Inverse rendering is a technique that utilizes machine learning to optimize (learn) a scene. In the next few challenges you will be implement features that allow us to understand material information as well as geometric information based off of a given 2D image and camera views. Firstly, our task is to understand the appearance of Zepelli's steel balls so the inverse renderer will begin with optimizing for material information.

## Application Flow For Material Optimization
To begin we will have initial guesses for things like objects albedo, and their metallic and roughness coefficients. Initial guesses for all objects must be different otherwise issues occur when trying to differentiate between them during the optimization loop.

Our optimization loop has four phases:
1. Phase 0 Emission Only: We select a subset of camera views and for each view we run the function ```render_and_grad_kernel```. This kernel shoots rays from the camera through every pixel, traces paths through the scene with NEE and GGX BRDF, writes the rendered pixel color to image[x, y], and accumulates an analytical emission gradient. The emission gradient uses the ratio trick: since radiance is proportional to emission, ```∂rad/∂emission ≈ rad/emission``` per channel. After rendering, compute_loss_view computes the log-space L2 loss between the rendered image and the target: ```sum of (log(1+rendered) - log(1+target))² / num_pixels```. The log compression gives dark pixels (where wall colors show via color bleeding) equal weight to bright pixels (near the light).
Then ```adam_step_emission``` updates only the emission using Adam. The gradient goes into the first moment ```m = 0.9*m + 0.1*g``` and second moment ```v = 0.999*v + 0.001*g²```. The bias-corrected update is ```emission -= lr * m̂ / (√v̂ + ε)```. Nothing else changes.

2. Phase 1 Albedo: Same render + loss as before, but now albedo is also updated. The analytical albedo gradient from ```render_and_grad_kernel``` is intentionally discarded (zeroed) because it pushes all materials toward the same value. Instead, per-material SPSA runs for every material.\
For each of the 7 materials, the SPSA function:\
Step 1: Saves the current albedo\
Step 2: Generates a random ±1 perturbation per RGB channel\
Step 3: Adds ```+0.015 * delta``` to the albedo, renders all views at low SPP (SPP_FD=2), computes the ```loss → loss_plus```\
Step 4: Adds ```-0.015 * delta``` to the albedo, renders all views, computes ```loss → loss_minus```\
Step 5: Restores the original albedo\
Step 6: Computes the gradient per channel: ```(loss_plus - loss_minus) / (2 * 0.015 * delta[c])```\

This costs 14 extra renders per iteration (7 materials × 2 renders each), but each gradient cleanly isolates one material's effect on the image. The gradient is EMA-smoothed across iterations (30% new + 70% history) to reduce noise, then ```adam_step_albedo``` updates each material's albedo with Adam.
Emission also continues updating from the analytical gradient.

3. Phase 2 Full Optimization: Everything from phase 1 continues, plus roughness and metallic SPSA starts. Every fifth iteration:\
Step 1 Roughness SPSA: perturb all materials' roughness by ```±0.005 * random_signs```, render twice, estimate the gradient.\
Step 2 Metallic SPSA: same as step 1 but with a larger perturbation of ```±0.02``` (metallic has a weaker visual effect so bigger probe).\

Both gradients are EMA-smoothed, then ```adam_step_roughness_metallic``` updates them with Adam, using gradient clipping ```(±5.0)``` to prevent SPSA outliers from corrupting the optimizer state. The learning rate for roughness/metallic is 30% of the albedo LR.
At the roughness/metallic phase transition, the Adam state for these parameters is reset to zero so that stale moments from the previous phase don't cause a bad first step.
4. Phase 3 Fine Tuning: Same as Phase 2 but with reduced learning rates: albedo LR × 0.5, roughness/metallic LR × 0.15. SPSA weight for albedo EMA blending drops from 0.5 to 0.3. This lets the optimizer make smaller corrections near convergence without oscillating.

After a training loop Epoch (10 iterations) we then render our guessed framebuffer to a window and compare it to the ground truth.

Other things to note before we actually begin coding again is what does ```render_and_grad_kernel``` even do? Well it renders a pixel like we normally do with a path tracing kernel we made in challenge 3, but instead of storing the color and moving on we compute the log-space loss gradient ```dL/dp``` with the chain rule through the log transform. Also for every sample we accumulate the emission gradient using ```rad/emission * dL/dp```. Once we have this loss we can determine safe parameters, if our guesses are too insane we can rollback to safe parameters by comparing loss spikage based on the difference of previous loss results.

## Note
Go to the file ```diffpt/kernels.py``` and fill in the functions with your solutions for the most part from the previous section.

## Challenge 4
Exhausted after only for the first stage of the race... Good thing the next act comes out in 2027! In the mean time lets learn something golden about inverse rendering rather than spin for a bit. For your first challenge in the inverse rendering section you will implement the function ```_render_all_views_loss_fd``` in the file ```train.py```. This function renders the scene from every camera view into the finite-distance image buffer, computes the per-view loss against the target images, and returns the total accumulated loss.

1. Loop over all the views
2. set the current active view
3. clear the FD image buffer by calling ```image_fd.fill(0.0)```
4. call ```render_fd_kernel```
5. Reset the FD loss accumulator, ```loss_fd[None] = 0.0``` before next step.
6. Compute per-view loss by calling ```compute_loss_fd_view```.
7. Accumulate the loss by adding loss_fd[None] to a total
8. Finally once all views have been rendered and the loss has been computed return the total from step 7.

### Challenge 5
The next couple of functions implement something called Simultaneous Perturbation Stochatic Approximation or SPSA to estimate the full gradient vector without having to fully test for it.

The first function using SPSA that we need to implement is in ```train.py``` called ```compute_roughness_metallic_grads```. This function estimates the gradients for roughness and metallic parameters using SPSA. Since these are scalar values per material SPSA can perturb them together in the range of plus or minus 1 directions.

1. Choose perturbation sizes, use FD_EPS as the base. Roughness and metallic can use the same or different scales, however metallic should have a higher since it is more sensitive.
2. Generate random perturbation directions. We can do this my generating two random values from [-1,1]: one for roughness and one for metallic then use something like numpy random choice. We do this for every material.
3. Before we apply our directions make sure to save the current roughness and metallic values for every material. I'd recommend storing a local list of size number of materials for each of them.
4. Apply the positive directions first and clamp each channel of RGB to [0.01,0.99] to avoid degenerate brdfs. Example for applying the directions is: ```roughness[i] = orig_roughness[i]+pert_size_rough*delta_r[i]``` Where delta_r is the directions and pert_size_rough is the initialized value from step 1.
5. Compute the positive loss, ```loss_plus = _render_all_views_loss_fd(info)```.
6. Apply the negative directions, same as step 4 except you subtract.
7. Compute the negative loss, ```loss_minus = _render_all_views_loss_fd(info)```.
8. Restore the original values for roughness and metallic
9. Finally compute the SPSA gradient estimates. Example for computing them is ```roughness_grad[i] = (loss_plus-loss_minus)/(2*pert_size_rough*delta_r[i])```.

You can refer to my implementation of ```compute_spsa_gradients``` in ```diffpt/geo_optimizer.py```.

### Challenge 6
The next SPSA function is ```compute_albedo_spsa_single_material``` located in ```train.py```. This function estimates the gradient for a single material's albedo (RGB, 3 Components), we do one material at a time so that the gradient signal is cleaner rather than blended with other possibly louder materials.

To implement this function we follow a similar approach to challenge 5:

1. Set perturbation size, FD_EPS is fine, but note that our albedo's values range from [0,1] if you wanted to use another value, for now we will call this c.
2. Generate a 3-component random direction using numpy random choice with the size of 3, this will be our delta.
3. save the original albedo before applying the direction.
4. Apply positive perturbation, example ```albedo[mat_idx] = [orig[0]+c* delta[0],orig[1]+c*delta[1],orig[2]+c*delta[2]]```.
5. Compute positive loss the same way as challenge 5
6. Apply negative perturbation, same as step 4 but subtraction.
7. Compute negative loss the same way as challenge 5
8. Restore original albedo
9. Compute the per-channel gradient, we will compute three grad_k values like so: ```grad_k = (loss_plus-loss_minus)/(2*c*delta[k])```. Then write back the result, ```albedo_grad[mat_idx] = [grad_0,grad_1,grad_2]```.

### Challenge 7
The next challenge is focused on optimizing for geometry by this point if you run ```uv run train.py --scenes/cornellbox.json``` you should be able to train for materials. Our optimizer of choice to update our hyperparameters is Adam, so of course challenge 7 is implementing our very own adam step kernel!!!

The function to implement is found in ```diffpt/geo_optimizer.py``` called ```_adam_step_kernel```. This function implements a singler step of the Adam optimizer for sphere geometry parameters specifically. Adam maintains per-parameter exponential moving averages of the gradient (first moment m) and squared gradient (second moment v), with bias correction.

1. Compute bias correction denominators, computed once outside the loop: ``` biasC1 = 1.0-beta1^t biasC2 = 1.0 - beta2^t```.
2. Loop over all the spheres and perform the next steps. Make sure to check if your current sphere exists by comparing, ```if i<ns:``` if this is try then break out of the loop.
3. Update the first moment, in this case we will update our center. To update the center we first need to get the center gradient. ```cg = sphere_center_grad_combined[i]``` then can update our moment, ```adam_m_center[i] = beta1 * adam_m_center[i] + (1-beta1) * cg```.
4. Update the second moment for our center, which is just an element-wise squared gradient: ```adam_v_center[i] = beta2 * adam_v_center[i] + (1 - beta2) * cg * cg```.
5. bias-correct and compute the stepping distance for the center:
```
  m_hat = adam_m_center[i] / betaC1
  v_hat = adam_v_center[i] / betaC2
  step = lr * m_hat / (sqrt(v_hat) + EPS)
  # lr is our learning rate
```
6. Write the center step, ```sphere_center_grad_combined[i] = step```.
7. Repeat steps 3 through 6 for radius, it is the same but variables will either say radius instead of center or have a ```_r``` at the end.

Overall, we love Adam because it combines the benefits of both momentum and RMSProp (the tracked running average of gradient magnitude). By diving the momentum smoothed gradient by the square root of the magnitude estimate we are able to automatically scale the learning rate per parameter. This decreases parameters with very large gradients from becoming too noisy instead their learning rate becomes much smaller preventing oscillation.
