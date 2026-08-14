export function ProgressBar({ progress }: { progress: number }) {
  return (
    <div
      style={{
        width: "460px",
        height: "4px",
        borderRadius: "10px",
        backgroundColor: "rgba(255,255,255,0.3)",
      }}
    >
      <div
        style={{
          width: `${progress}%`,
          height: "100%",
          backgroundColor: "#fff",
          borderRadius: "10px",
          transition: "width 0.5s ease",
        }}
      ></div>
    </div>
  );
}
