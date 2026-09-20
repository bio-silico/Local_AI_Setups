# Local AI Setups — GeneMolX AI LLC

> **DRAFT — under review.**

## Vision

Two years of building a complete, personalized **local AI framework** on a private workstation.

This repository shares **the foundation** of it — the machine, the GPU and the local AI models —
so others can build the same kind of system and use AI the way GeneMolX uses it: private,
local, under your own control.

## What this repository is

- A guide to set up the foundation of a local AI system: machine, GPU and memory, and local
  AI models with Ollama and LM Studio
- Setup steps, configuration examples and lessons learned, with the GeneMolX system as the
  worked example

## What this repository is not

- **Not GeneMolX's data** — no proprietary data, documents or knowledge bases
- **Not GeneMolX's information** — no internal records, customers or business content
- **Not GeneMolX's connections** — no credentials, keys, hostnames, endpoints or live configs
- Not a copy of the running GeneMolX system

## Target users

People who want **local AI for privacy** — their data stays on their own machine — and who
have, or are buying, one machine for it.

This repository uses the GeneMolX infrastructure as a **template**, so they can set up and use
AI the way GeneMolX does.

**The machine decides how big a model you can run** — see [Requirements](#requirements).

**The GeneMolX series.** This repository is the foundation. Each next layer gets its own
repository:

- **Local AI foundation** — machine, GPU, local models *(this repository)*
- A database "brain" — knowledge, memory and data (Supabase)
- AI agents — n8n
- Working with your data through RAG and memory — Open WebUI
- Exploring and querying your databases — LangChain (Python)

Need something custom? GeneMolX AI LLC helps with individual setups — see [genemolx.com](https://www.genemolx.com).

## Disclaimer

- **Template, not a product.** Provided as is, without warranty. Test it on your own system.
- **Your data, your responsibility.** Local AI is only as private and secure as your own
  setup — passwords, access, backups and network are yours to manage.
- **No GeneMolX data.** Every example, config and dataset here is a template or mock data.
- **Third-party software and models have their own licences** (Ollama, LM Studio, GPU
  drivers, each model). Check them before any commercial use.

## Requirements

Three levels, one machine each — no cluster, no cloud.

| Level | Machine | Models |
|---|---|---|
| **Basic** | the GeneMolX reference system (below) | 7B–14B |
| **Developer** | a workstation with a 32 GB GPU | ~32B |
| **High-end** | a machine with 128 GB unified memory | 70B, up to ~120B |

### Basic — the GeneMolX reference system

| Component | Specification |
|---|---|
| Operating system | Ubuntu 24.04 LTS |
| Processor | Intel Core i9-12900K — 16 cores / 24 threads |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA GeForce RTX 4070 — 12 GB VRAM |
| Storage | NVMe SSD — 1 TB system + 4 TB data and models |
| AI runtimes | Ollama and LM Studio |

Runs 7B–14B models fully on the GPU, at roughly 35 tokens per second. Larger models (20–27B)
still run, split between GPU and system RAM, but slower.

### Example machines — Developer and High-end

| Level | Machine | Memory for models | Notes |
|---|---|---|---|
| Developer | Workstation + NVIDIA RTX 5090 | 32 GB VRAM + system RAM | fastest for ≤ 32B, ≈ 71 tok/s on 32B |
| High-end | Apple Mac Studio M5 Max / M5 Ultra | up to 128 GB unified (M5 Max) | Ultra configurable higher |
| High-end | NVIDIA DGX Spark (GB10) | 128 GB unified | 32B ≈ 38 tok/s · 70B Q4 ≈ 22 tok/s |
| High-end | AMD Ryzen AI Max+ 395 mini PC | 128 GB unified | 32B fine · 70B slow (≈ 5 tok/s) |

Specs as of September 2026.

**What decides the level:**
- **Memory size** decides *how big* a model fits — GPU VRAM, or unified memory shared by CPU and GPU.
- **Memory bandwidth** decides *how fast* it answers. A 32 GB GPU is the fastest up to 32B;
  128 GB unified-memory machines trade speed for room to run 70B and larger.

## Step 1 — Organize your AI directory

Before installing anything, create **one home for your AI system**, so you always know where
your models, configs and notes live.

```bash
mkdir ~/System-AI
cd ~/System-AI
mkdir Ollama LM-Studio      # one folder per component, added as you go
```

**Best location:** a dedicated fast drive (NVMe SSD) for AI, separate from the system drive —
models grow fast. On the GeneMolX reference system, Docker data and AI configs live on a second
NVMe drive; Ollama and LM Studio still keep their models in their default folders on the
system drive. Step 3 shows how to point both to `System-AI/`.

**Programs install system-wide; `System-AI/` holds what is yours** — models, configs, notes
and backups.

**There is always more than one way to install.** Each component can run natively (installer
or command line), as a desktop app, or in Docker. This guide shows **how GeneMolX runs it** as
the worked example, and notes the other options.

## Step 2 — GPU and memory

The GPU — and the memory it can use — decides **how big** a model you can run and **how fast**
it answers. Set it up and check it before installing Ollama or LM Studio.

Two kinds of machines:

- **Discrete GPU** (NVIDIA, AMD Radeon) — the model lives in the GPU's own memory (VRAM).
  What doesn't fit spills to system RAM and runs much slower.
- **Unified memory** (Apple Silicon, AMD Ryzen AI Max, NVIDIA DGX Spark) — CPU and GPU share
  one large memory pool. Bigger models fit; speed depends on memory bandwidth.

### Worked example — GeneMolX (NVIDIA, Ubuntu)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 — 12 GB VRAM |
| Driver | NVIDIA 580 (Ubuntu package), reports CUDA 13.0 |
| Docker GPU access | NVIDIA Container Toolkit, Docker default runtime `nvidia` |
| CUDA toolkit | **not installed — not needed.** Ollama and LM Studio ship their own CUDA runtime |

**Check that models really run on the GPU:**

```bash
nvidia-smi                      # GPU, driver, and which process holds GPU memory
ollama ps                       # Ollama: loaded models, GPU vs CPU share
lms ps                          # LM Studio: loaded models
journalctl -u ollama | grep "layers to GPU"   # e.g. "offloaded 33/33 layers to GPU"
```

`offloaded 0/33 layers to GPU` means the model ran entirely on the CPU — no error, just slow.

**Lessons learned on this system:**

1. **One GPU, two runtimes.** Ollama and LM Studio share the same GPU memory, and whichever
   loads a model first keeps it. A model loaded by hand in LM Studio, with no idle timeout,
   held ~10 GB for about three weeks — and Ollama silently fell back to the CPU.

   LM Studio's idle timeout (`JIT model TTL`) applies **only to models loaded by a request**,
   not to models loaded by hand.

   | | Ollama loading llama3 (8B) |
   |---|---|
   | Before | `offloaded 0/33 layers to GPU` — ran on the CPU |
   | After | `offloaded 33/33 layers to GPU` — 100% GPU |

   **Policy chosen — models load on request only:**
   - a request loads the model; it unloads after 1 hour idle (`JIT model TTL = 3600 s`)
   - no manual loads for everyday use; if one is needed, give it a timeout:
     `lms load <model> -c <context> --ttl 3600`
   - save each model's context length as its default, so an automatic reload keeps it
   - trade-off: the first request after idle waits for the model to load (~10 s for a 14B);
     a shorter timeout frees the GPU sooner but means more of these cold starts

   **Who holds the GPU right now** — one line:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; lms ps; ollama ps
   ```

   Models bigger than the GPU (13–17 GB on a 12 GB card) never run 100% on the GPU — they
   always split with system RAM. Freeing the GPU moves them from zero GPU layers to partial.

   **With one GPU, expect this:** only one model fits in GPU memory at a time, so a request for
   a different model swaps it (~6–10 s), and requests are queued rather than answered in
   parallel. Plan which model is the everyday one.

2. **More context in the same memory — quantize the KV cache.** The context (everything the
   model is "reading") is held in GPU memory next to the model. Stored in 8-bit instead of the
   default 16-bit, it takes about half the space, with no measurable quality loss.

   On this system, with the same 12 GB card:

   | Model | Context before | Context after |
   |---|---|---|
   | qwen3-14b | 12,288 tokens | **24,576** |
   | gemma-4-12b | 12,288 tokens | **49,152** |

   Two settings did it: 8-bit KV cache, and **one prediction slot per model** instead of four —
   slots split the context between simultaneous requests, so fewer slots means more context each.

   - **LM Studio:** KV cache type and slots are in the model's load settings
   - **Ollama:** `OLLAMA_KV_CACHE_TYPE=q8_0` (or `q4_0` for a quarter of the memory, with some
     quality loss) and `OLLAMA_NUM_PARALLEL` (Step 3)

   Worth doing when prompts fail for exceeding the context, before buying a bigger GPU.

3. **Kernel upgrades — install kernel headers first.** On Ubuntu the NVIDIA modules are built
   at install time and need the matching headers. Without them the driver is missing after
   reboot and Ollama silently runs on the CPU. Install `linux-headers-<version>` before the new
   kernel, and check `nvidia-smi` after rebooting.

### NVIDIA — Linux

- **GPU:** compute capability 5.0 or newer
- **Driver:** 550 or newer (570+ for compute capability 5.0–6.2)
- **LM Studio:** select the CUDA engine in *Settings → Runtime*
- **Docker containers that need the GPU:** install the NVIDIA Container Toolkit, then
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- **Several GPUs:** choose which ones Ollama uses with `CUDA_VISIBLE_DEVICES`

### Apple Silicon — macOS

- **Nothing to install.** GPU acceleration uses Metal, built into macOS.
- **LM Studio:** Metal and **MLX** engines — MLX is Apple's framework, often the fastest on a Mac.
- **Memory:** by default macOS lets the GPU use roughly 75% of unified memory. It can be raised:
  ```bash
  sudo sysctl iogpu.wired_limit_mb=<MB>     # e.g. 57344 = 56 GB on a 64 GB Mac
  ```
  It is a ceiling, not a reservation. Leave room for macOS (about 8 GB on 64 GB, 14 GB on
  128 GB) — too high and the system becomes unstable. The setting resets at reboot.

### AMD — Linux

**Radeon graphics cards**

- **Ollama:** needs the AMD **ROCm v7** driver. Supported: Radeon RX 9000 / 7000 series,
  Radeon PRO, Radeon AI PRO, Instinct.
- **LM Studio:** Vulkan engine; some cards on Linux can use the faster ROCm engine.
- **Vulkan** is a fallback when ROCm doesn't support a card.
- **Several GPUs:** `ROCR_VISIBLE_DEVICES`

**Ryzen AI Max+ 395 (unified memory, up to 128 GB)**

- Supported by Ollama (ROCm target `gfx1151`).
- To let the GPU use most of the 128 GB on Linux:
  - BIOS: set the *UMA frame buffer* to the minimum (e.g. 512 MB)
  - kernel parameters `amdgpu.gttsize` and `ttm.pages_limit` raise the GPU's share
    (commonly ~96–120 GB)
- If ROCm is unstable, the Vulkan backend is the usual fallback.

### References

- [Ollama — hardware support](https://docs.ollama.com/gpu)
- [NVIDIA Container Toolkit — installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [CUDA GPUs — compute capability](https://developer.nvidia.com/cuda-gpus)
- [AMD ROCm — Radeon and Ryzen on Linux](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- [Apple Silicon — raising the GPU memory limit](https://modelpiper.com/blog/iogpu-wired-limit-mb-mac)
- [Ryzen AI Max — GTT memory on Linux](https://craftrigs.com/articles/amd-ryzen-ai-max-linux-vram-gtt-memory-guide/)

## Step 3 — Local AI models: installation and setup

Your machine and GPU are ready; now the models. You download and run local AI models with
**Ollama** and **LM Studio**. Both serve the models through an API on your machine, so other
tools can use them.

GeneMolX uses **both**. That is good practice — each has its strengths — as long as they share
the GPU well (Step 2, lesson 1).

| | Ollama | LM Studio |
|---|---|---|
| What it is | background service + command line | desktop app + `lms` command line (also headless) |
| API port | 11434 | 1234 (OpenAI-compatible) |
| Linux | install script, runs as a system service | `.deb`, AppImage, or headless `llmster` |
| macOS / Windows | desktop app | desktop app |
| Docker | official image `ollama/ollama` | — |

### Ollama

**Install**

- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh` — creates the `ollama` user and a
  systemd service that starts at boot. Manual install from a tarball is also possible.
- **macOS / Windows:** download the app from ollama.com.
- **Docker:** `ollama/ollama` image — GPU access needs the NVIDIA Container Toolkit (Step 2).

**Worked example — GeneMolX (Ubuntu)**

| | |
|---|---|
| Installed with | the official install script |
| Runs as | systemd service `ollama`, user `ollama`, starts at boot |
| Changed settings | one: `OLLAMA_HOST=0.0.0.0` — open on the local network (see Principles) |
| Models stored in | the default location, `/usr/share/ollama/.ollama/models` |

**Change settings** — Linux:

```bash
sudo systemctl edit ollama        # opens an override file
```
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_MODELS=/path/to/System-AI/Ollama/models"
```
```bash
sudo systemctl restart ollama
```

macOS: `launchctl setenv <NAME> <value>`, then restart the app. Windows: set the variable in
*Environment Variables*, then restart Ollama.

**Settings that matter**

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_HOST` | `127.0.0.1:11434` | who can reach Ollama — local only by default |
| `OLLAMA_MODELS` | Linux `/usr/share/ollama/.ollama/models` · macOS / Windows `~/.ollama/models` | where models are stored. On Linux the `ollama` user must own the new folder: `sudo chown -R ollama:ollama <folder>` |
| `OLLAMA_KEEP_ALIVE` | 5 minutes | how long an idle model stays in memory |
| `OLLAMA_CONTEXT_LENGTH` | 4096 tokens | default context window |
| `OLLAMA_FLASH_ATTENTION` | automatic | `1` forces it on — less memory for long contexts |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | `q8_0` halves context memory, `q4_0` quarters it (small quality loss) — see Step 2, lesson 2 |
| `OLLAMA_MAX_LOADED_MODELS` | 3 per GPU | how many models can be loaded at once |
| `OLLAMA_NUM_PARALLEL` | 1 | parallel requests per model |

**Everyday commands**

```bash
ollama pull <model>      # download
ollama run <model>       # chat in the terminal
ollama list              # downloaded models
ollama ps                # loaded models — PROCESSOR shows 100% GPU / CPU / split
ollama stop <model>      # unload now
journalctl -e -u ollama  # service log (Linux)
```

### LM Studio

**Install**

- **Desktop app** (Linux `.deb` or AppImage, macOS, Windows): download from lmstudio.ai.
  On Ubuntu 22.04+ the AppImage needs `libfuse2`.
- **Headless — `llmster` daemon, no GUI** (servers, always-on machines):
  `curl -fsSL https://lmstudio.ai/install.sh | bash`, then `lms daemon up` and `lms server start`.
- **Engines:** pick the one for your GPU in *Settings → Runtime* — CUDA, Vulkan, ROCm, Metal,
  MLX (Step 2).

**Worked example — GeneMolX (Ubuntu)**

| | |
|---|---|
| Installed with | `.deb` package → `/opt/LM-Studio` |
| Runs as | desktop app, starts minimized at login |
| Engine | CUDA (llama.cpp) |
| API server | port 1234, open on the local network |
| Model loading | on request (JIT), unloaded after 1 hour idle (Step 2, lesson 1) |
| Context | raised per model with 8-bit KV cache: 24,576 for qwen3-14b, 49,152 for gemma-4-12b (Step 2, lesson 2) |
| Prediction slots | 1 per model, so one request gets the whole context |
| Speed | ≈ 35 tokens/s on 12–14B models |
| Models stored in | `~/.lmstudio/models` (the default) |

**Settings that matter**

| Setting | Where | What it does |
|---|---|---|
| Server port and network | Developer → server settings | local only or local network |
| Just-in-time (JIT) loading | Developer → server settings | a request loads the model it needs |
| Idle TTL (`JIT model TTL`) | Developer → server settings | unloads request-loaded models after idle time |
| Default context length | Settings | context for new loads; can be saved per model |
| KV cache quantization | model load settings | 8-bit instead of 16-bit — much more context in the same memory (Step 2, lesson 2) |
| Prediction slots | model load settings | how many requests share the context. 1 slot = full context for one request |
| Models folder | My Models | where models are stored — move it to `System-AI/LM-Studio` (Step 1) |
| Run server on login | Settings | keeps the API up without opening the window |

**Everyday commands**

```bash
lms ls                                   # downloaded models
lms ps                                   # loaded models
lms get <model>                          # download
lms load <model> -c <context> --ttl 3600 # load, with an idle timeout
lms unload <model>                       # unload now
lms server start | stop | status         # API server
```

### References

- [Ollama — Linux install](https://docs.ollama.com/linux)
- [Ollama — FAQ and settings](https://docs.ollama.com/faq)
- [LM Studio — download](https://lmstudio.ai/download)
- [LM Studio — run as a service (headless)](https://lmstudio.ai/docs/developer/core/headless)

## Principles

- Everything runs locally — no data leaves the machine unless you choose
- Reproducible — someone else can follow it and get a working system
- Templates and examples only — every file reviewed before publishing
- **Network access is a deliberate choice, made for functionality.** Each service can be:
  - **local only** — this machine
  - **local network** — your other machines use it (e.g. another system calling Ollama, n8n credentials pointing at a port)
  - **internet** — reached from outside

  The GeneMolX system opens services on the local network on purpose, so its machines work
  together. Opening a service is fine when you need it — know which ones are open, why, and
  set them up carefully. External use and connections need extra care.

  **How to check what is really exposed** — lesson learned on the GeneMolX system:
  - **Check the firewall's real state, not only whether its service runs.** On Ubuntu,
    `systemctl` can report `ufw` as *active* while the firewall is off. Use `sudo ufw status`
    (or `ENABLED=` in `/etc/ufw/ufw.conf`).
  - **Include IPv6.** Many home connections give the machine public IPv6 addresses, and
    services listening on all interfaces listen on IPv6 too.
  - **Test from outside.** A phone on mobile data (Wi-Fi off) trying to open the service's
    address shows whether it is reachable from the internet.
  - **Check before changing.** Turning on a firewall affects every service on the machine, not
    just the one you meant to protect. A fix built on an unchecked assumption can be riskier
    than the problem.

## Open questions

- Security review per service — in each repository of the series

## Licence

© 2026 GeneMolX AI LLC — licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Full text in [LICENSE](LICENSE).

You may share and adapt this guide, including commercially, as long as you give credit:

> *Local AI Setups* by GeneMolX AI LLC — https://github.com/bio-silico/Local_AI_Setups — CC BY 4.0

This licence covers this repository only. Ollama, LM Studio, GPU drivers and each model keep
their own licences (see [Disclaimer](#disclaimer)).
