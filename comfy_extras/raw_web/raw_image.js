const app = window.comfyAPI?.app?.app;
const api = window.comfyAPI?.api?.api;

if (app && api) {
    // Listen for ALL events to find the right one
    const origDispatch = api.dispatchEvent.bind(api);
    api.dispatchEvent = function(event) {
        if (event.type !== "status" && event.type !== "progress") {
            console.log("[raw_image] api event:", event.type, event.detail ? JSON.stringify(event.detail).substring(0, 200) : "");
        }
        return origDispatch(event);
    };

    console.log("[raw_image] monitoring api events, api type:", typeof api, api.constructor?.name);
}
