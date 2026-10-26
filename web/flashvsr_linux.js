// Shows the video written by "FlashVSR Video Combine (Linux)" inside the node.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FlashVSRLinux.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FlashVSRLinuxVideoCombine") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const item = message?.flashvsr_video?.[0];
            if (!item) return;
            const params = new URLSearchParams({
                filename: item.filename, subfolder: item.subfolder ?? "", type: item.type ?? "output",
                t: String(Date.now()),
            });
            if (!this.flashvsrVideo) {
                const el = document.createElement("video");
                el.controls = true;
                el.loop = true;
                el.muted = true;
                el.autoplay = true;
                el.style.width = "100%";
                this.flashvsrVideo = el;
                this.addDOMWidget("preview", "flashvsr_video", el, { serialize: false });
            }
            this.flashvsrVideo.src = api.apiURL("/view?" + params.toString());
        };
    },
});
