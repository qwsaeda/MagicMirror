import banner from "@/assets/images/magic-mirror.svg";
import { useDownload } from "@/hooks/useDownload";
import { useServer } from "@/hooks/useServer";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

export function LaunchPage() {
  const { t } = useTranslation();
  const { download } = useDownload();
  const { launch } = useServer();
  const navigate = useNavigate();

  useEffect(() => {
    download().then((ok) => {
      if (ok) {
        launch().then((success) => {
          if (success) {
            navigate("/mirror");
          }
        });
      }
    });
  }, []);

  return (
    <div
      data-tauri-drag-region
      style={{
        border: "1px solid rgba(0, 0, 0, 0.1)",
        boxShadow: "0 4px 10px rgba(0, 0, 0, 0.3), 0 8px 20px rgba(0, 0, 0, 0.3)",
      }}
      className="w-540px h-320px bg-#151515 color-white flex-col-c-c gap-8px p-10px"
    >
      <img src={banner} className="w-80% object-cover cursor-default pointer-events-none select-none" />
      <p>{t("Starting... First load may take longer, please wait.")}</p>
    </div>
  );
}
