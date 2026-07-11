# ✨ CUDA learning plan: Mandelbrot renderer

## 🌈 A beautiful first visual CUDA project
This is a wonderful project because you get immediate visual feedback from your GPU work.

> You will learn how to map a 2D problem onto many threads and generate a striking image.

### 🧭 Quick mission
Build a program that renders a Mandelbrot or Julia set image using CUDA.

### 🏁 What success looks like
- You understand how to map image coordinates to GPU threads
- You can explain how the iteration loop is parallelized
- You produce an image as output

---

## 1. 🧠 Start with the math
Before writing CUDA code, understand the basic iteration rule.

### What to focus on
- complex numbers
- escape condition
- iteration count

### Why this helps
The math is simple enough to follow while still giving a rich visual result.

---

## 2. 🧪 Build a CPU reference version
Write the renderer first in a normal loop.

### What to do
- Choose a region of the complex plane
- Compute pixel colors based on iteration count
- Save or display the image

---

## 3. 🧵 Think in terms of pixels
A natural mapping is:
- One thread handles one pixel

### Key idea
Each pixel can be computed independently, making this ideal for GPU execution.

---

## 4. ⚙️ Design the kernel conceptually
Each thread should:
- Determine its pixel location
- Convert it to a complex coordinate
- Run the iteration loop
- Store the result as a color value

### Big idea
You are turning an image into a large set of independent computations.

---

## 5. 🧩 Think about output layout
This project teaches you how output arrays are organized.

### Important hints
- Image data is often stored as a flat buffer
- Color channels matter
- A simple color mapping makes the result easier to interpret

---

## 6. ✅ Verify correctness
Check the output image visually.

### Check
- Does the shape look like a Mandelbrot set?
- Are the colors reasonable?

---

## 7. ⏱️ Measure performance
Once it works, compare CPU and GPU speed at larger resolutions.

### Why this matters
This is a very intuitive way to see how GPU parallelism helps on structured workloads.

---

## 8. 🌟 Stretch ideas
When you are comfortable, try:
- Julia sets
- zooming controls
- color palette variations
- higher resolution images

---

## 💡 Helpful hints while you write it manually
- Start with a small resolution first
- Keep the iteration limit modest
- Focus on one pixel-to-thread mapping
- Use the output image as a debugging tool

---

## 🎓 What you will learn from this project
You will build intuition for:
- 2D parallel mapping
- output buffers
- visual debugging
- structured GPU workloads
