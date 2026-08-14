import { useCallback, useEffect } from "react";
import { useXState } from "xsta";
import { Server } from "../services/server";

const kSwapFaceRefs: {
  id: number;
  cancel?: VoidFunction;
} = {
  id: 1,
  cancel: undefined,
};

export function useSwapFace() {
  const [isSwapping, setIsSwapping] = useXState("isSwapping", false);
  const [output, setOutput] = useXState<string | null>("swapOutput", null);

  const swapFace = useCallback(
    async (inputImage: string, targetFace: string) => {
      await kSwapFaceRefs.cancel?.();
      setIsSwapping(true);
      const taskId = (kSwapFaceRefs.id++).toString();
      kSwapFaceRefs.cancel = async () => {
        const success = await Server.cancelTask(taskId);
        if (success) {
          setIsSwapping(false);
        }
      };
      const result = await Server.createTask({
        id: taskId,
        inputImage,
        targetFace,
      });
      kSwapFaceRefs.cancel = undefined;
      setOutput(result);
      setIsSwapping(false);
      return result;
    },
    []
  );

  useEffect(() => {
    return () => {
      kSwapFaceRefs.cancel?.();
    };
  }, []);

  return {
    isSwapping,
    output,
    swapFace,
    cancel: () => kSwapFaceRefs.cancel?.(),
  };
}
