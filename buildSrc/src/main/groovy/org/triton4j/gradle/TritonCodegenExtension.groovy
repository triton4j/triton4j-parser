package org.triton4j.gradle

import javax.inject.Inject

import org.gradle.api.Project
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.model.ObjectFactory
import org.gradle.api.provider.Property

class TritonCodegenExtension {
    final ConfigurableFileCollection inputs
    final DirectoryProperty outputDir
    final Property<String> packageName
    final Property<String> className
    final Property<String> comment
    final Property<String> language
    final Property<Boolean> verbose
    final Property<Boolean> continueOnError

    @Inject
    TritonCodegenExtension(Project project) {
        ObjectFactory objects = project.objects
        this.inputs = project.files()
        this.outputDir = objects.directoryProperty()
        this.packageName = objects.property(String)
        this.className = objects.property(String)
        this.comment = objects.property(String)
        this.language = objects.property(String)
        this.verbose = objects.property(Boolean)
        this.continueOnError = objects.property(Boolean)

        this.inputs.from(project.layout.projectDirectory.dir("tutorials_python"))
        this.outputDir.convention(project.layout.buildDirectory.dir("generated-triton-java"))
        this.packageName.convention("org.triton4j.triton.test")
        this.language.convention("tl")
        this.verbose.convention(false)
        this.continueOnError.convention(true)
    }
}
