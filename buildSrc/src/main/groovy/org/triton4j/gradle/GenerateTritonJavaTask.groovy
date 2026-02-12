package org.triton4j.gradle

import javax.inject.Inject

import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.process.ExecOperations

abstract class GenerateTritonJavaTask extends DefaultTask {
    @InputFiles
    final ConfigurableFileCollection pythonInputs = project.files()

    @OutputDirectory
    final DirectoryProperty outputDir = project.objects.directoryProperty()

    @Input
    final Property<String> packageName = project.objects.property(String)

    @Input
    final Property<String> language = project.objects.property(String)

    @Input
    final Property<Boolean> verbose = project.objects.property(Boolean)

    @Input
    final Property<Boolean> continueOnError = project.objects.property(Boolean)

    @Optional
    @Input
    final Property<String> className = project.objects.property(String)

    @Optional
    @Input
    final Property<String> comment = project.objects.property(String)

    @Inject
    protected abstract ExecOperations getExecOperations()

    @TaskAction
    void generate() {
        def args = ["generate"]
        pythonInputs.files.sort { a, b -> a.absolutePath <=> b.absolutePath }.each { file ->
            args << file.absolutePath
        }
        args.addAll(["-o", outputDir.get().asFile.absolutePath])
        args.addAll(["-p", packageName.get()])
        args.addAll(["-l", language.get()])
        if (className.present) {
            args.addAll(["-c", className.get()])
        }
        if (comment.present) {
            args.addAll(["--comment", comment.get()])
        }
        if (verbose.get()) {
            args << "-v"
        }
        if (continueOnError.get()) {
            args << "--continue-on-error"
        }

        def sourceSets = project.extensions.getByType(SourceSetContainer)
        def runtimeClasspath = sourceSets.named("main").get().runtimeClasspath

        logger.lifecycle("Running Triton code generation for {} input path(s) -> {}", pythonInputs.files.size(),
                outputDir.get().asFile)

        def command = ["java", "--add-modules", "jdk.incubator.code", "-cp", runtimeClasspath.asPath,
                "org.triton4j.cli.TritonParserCli"]
        command.addAll(args)

        execOperations.exec {
            commandLine(command)
        }
    }
}
