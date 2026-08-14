import { useCallback } from "react";
import { useXState } from "xsta";
import { Server, ServerStatus } from "../services/server";

const kStatusKey = "serverStatus";

export function useServer() {
  const [status, setStatus] = useXState<ServerStatus>(kStatusKey, "idle");

  const launch = async () => {
    if (status !== "idle") {
      return true;
    }

    setStatus("launching");
    const success = await Server.launch();
    
    if (success) {
      setStatus("running");
    } else {
      setStatus("idle");
    }
    return success;
  };

  const kill = useCallback(async () => {
    setStatus("idle");
    await Server.kill();
  }, []);

  return { status, launch, kill };
}
