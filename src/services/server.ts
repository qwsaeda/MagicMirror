import { invoke } from "@tauri-apps/api/core";

export type ServerStatus = "idle" | "launching" | "running";

export interface Task {
  id: string;
  inputImage: string;
  targetFace: string;
}

class _Server {
  _launched = false;
  _baseURL = "http://localhost:8023";

  async isDownloaded(): Promise<boolean> {
    try {
      const exeDir = await invoke<string>("get_exe_dir");
      return await invoke<boolean>("file_exists", { path: `${exeDir}\\server.exe` });
    } catch {
      return false;
    }
  }

  async download(): Promise<boolean> {
    return await this.isDownloaded();
  }

  async launch(): Promise<boolean> {
    if (this._launched) {
      return true;
    }

    try {
      // 始终调用 spawn_server：Rust 端统一处理接管/清理残留/启动
      const spawned = await invoke<boolean>("spawn_server");
      if (!spawned) {
        return false;
      }

      // 等待 server 就绪（最多 30 秒）
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 500));
        const running = await invoke<boolean>("is_server_running");
        if (running) {
          break;
        }
      }

      // 等待模型完全加载（GFPGAN 340MB 需要时间）
      await new Promise((r) => setTimeout(r, 15000));

      const prepared = await this.prepare();
      if (prepared) this._launched = true;
      return prepared;
    } catch {
      return false;
    }
  }

  async kill(): Promise<void> {
    this._launched = false;
    try {
      await invoke("kill_server");
    } catch (e) {
      console.error("Failed to kill server:", e);
    }
  }

  async status(): Promise<ServerStatus> {
    try {
      const res = await fetch(`${this._baseURL}/status`, {
        method: "get",
        signal: AbortSignal.timeout(5000),
      });
      const data = await res.json();
      return data.status || "idle";
    } catch {
      return "idle";
    }
  }

  async prepare(): Promise<boolean> {
    try {
      const res = await fetch(`${this._baseURL}/prepare`, {
        method: "post",
        signal: AbortSignal.timeout(180000),
      });
      const data = await res.json();
      return data.success === true;
    } catch {
      return false;
    }
  }

  async createTask(task: Task): Promise<string | null> {
    try {
      const res = await fetch(`${this._baseURL}/task`, {
        method: "post",
        headers: {
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify(task),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.result || null;
    } catch {
      return null;
    }
  }

  async cancelTask(taskId: string): Promise<boolean> {
    try {
      const res = await fetch(`${this._baseURL}/task/${taskId}`, {
        method: "delete",
      });
      const data = await res.json();
      return data.success === true;
    } catch {
      return false;
    }
  }
}

export const Server = new _Server();
