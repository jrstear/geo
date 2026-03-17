// SystemJS v0.21 AMD detection regex doesn't match arrow-function factories.
// Explicitly set format so window.define is populated before evaluation.
PluginsAPI.SystemJS.config({
    meta: {
        "GCPSighter/build/TaskView.js": { format: "amd" }
    }
});

PluginsAPI.Dashboard.addTaskActionButton(
    ["GCPSighter/build/TaskView.js"],
    function(args, TaskView) {
        return React.createElement(TaskView, {
            task: args.task,
            apiURL: "/api/plugins/GCPSighter"
        });
    }
);
