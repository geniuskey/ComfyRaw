const app = window.comfyAPI?.app?.app;
const api = window.comfyAPI?.api?.api;

if (app && api) {
    // Enumerate api to find event mechanism
    console.log("[raw_image] api constructor:", api.constructor?.name);
    console.log("[raw_image] api proto methods:", Object.getOwnPropertyNames(Object.getPrototypeOf(api)).filter(k => typeof api[k] === "function"));

    // Try monkey-patching dispatchCustomEvent if it exists
    if (typeof api.dispatchCustomEvent === "function") {
        const orig = api.dispatchCustomEvent.bind(api);
        api.dispatchCustomEvent = function (type, data) {
            if (type === "executed" || type === "executing") {
                console.log("[raw_image] customEvent:", type, JSON.stringify(data)?.substring(0, 300));
            }
            return orig(type, data);
        };
        console.log("[raw_image] patched dispatchCustomEvent");
    }

    // Also try standard addEventListener
    api.addEventListener("executed", ({ detail }) => {
        console.log("[raw_image] std executed event:", JSON.stringify(detail)?.substring(0, 300));
    });

    console.log("[raw_image] ready");
}
