package org.triton4j.gradle

import org.gradle.api.Plugin
import org.gradle.api.Project

class TritonCodegenPlugin implements Plugin<Project> {
    @Override
    void apply(Project project) {
        def extension = project.extensions.create("tritonCodegen", TritonCodegenExtension, project)

        project.pluginManager.withPlugin("java") {
            def generateTask = project.tasks.register("generateTritonJava", GenerateTritonJavaTask) { task ->
                group = "code generation"
                description = "Generate Java code from Triton Python files using TritonParser CLI."

                dependsOn(project.tasks.named("classes"))

                pythonInputs.from(extension.inputs)
                outputDir.set(extension.outputDir)
                packageName.convention(extension.packageName)
                className.convention(extension.className)
                comment.convention(extension.comment)
                language.convention(extension.language)
                verbose.convention(extension.verbose)
                continueOnError.convention(extension.continueOnError)
            }

            project.tasks.named("build").configure {
                dependsOn(generateTask)
            }
        }
    }
}
