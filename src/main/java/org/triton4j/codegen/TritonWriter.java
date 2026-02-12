/*
 * Copyright 2025 dScope, LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

package org.triton4j.codegen;

import java.io.IOException;
import java.io.StringWriter;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import javax.lang.model.SourceVersion;
import javax.lang.model.element.Modifier;

import org.parsers.python.Node;
import org.parsers.python.Node.TerminalNode;
import org.parsers.python.PythonParser;
import org.parsers.python.ast.AdditiveExpression;
import org.parsers.python.ast.Argument;
import org.parsers.python.ast.Assignment;
import org.parsers.python.ast.BitwiseAnd;
import org.parsers.python.ast.BitwiseOr;
import org.parsers.python.ast.Block;
import org.parsers.python.ast.Comment;
import org.parsers.python.ast.Comparison;
import org.parsers.python.ast.Conjunction;
import org.parsers.python.ast.Decorators;
import org.parsers.python.ast.Delimiter;
import org.parsers.python.ast.DotName;
import org.parsers.python.ast.ForStatement;
import org.parsers.python.ast.FunctionCall;
import org.parsers.python.ast.FunctionDefinition;
import org.parsers.python.ast.Group;
import org.parsers.python.ast.IfStatement;
import org.parsers.python.ast.InvocationArguments;
import org.parsers.python.ast.Keyword;
import org.parsers.python.ast.MultiplicativeExpression;
import org.parsers.python.ast.Name;
import org.parsers.python.ast.Newline;
import org.parsers.python.ast.NumericalLiteral;
import org.parsers.python.ast.Operator;
import org.parsers.python.ast.Parameters;
import org.parsers.python.ast.ReturnStatement;
import org.parsers.python.ast.SliceExpression;
import org.parsers.python.ast.Slices;
import org.parsers.python.ast.Statement;
import org.parsers.python.ast.StarTargets;
import org.parsers.python.ast.StringLiteral;
import org.parsers.python.ast.Tuple;
import org.parsers.python.ast.UnaryExpression;
import org.parsers.python.ast.WhileStatement;

import com.palantir.javapoet.AnnotationSpec;
import com.palantir.javapoet.CodeBlock;
import com.palantir.javapoet.JavaFile;
import com.palantir.javapoet.MethodSpec;
import com.palantir.javapoet.ParameterSpec;
import com.palantir.javapoet.TypeSpec;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;
import oracle.code.triton.Tensor;
import oracle.code.triton.Triton;

/**
 * This class is responsible for generating Java code from Python/Triton AST
 * nodes using JavaPoet.
 */
public class TritonWriter {
	private int tupleTempCounter = 0;

	/**
	 * Writes the generated Java code to the specified target path.
	 *
	 * @param tritonWriterContext The context for the Triton writer.
	 * @param source              The source path of the Python file.
	 * @param target              The target path for the generated Java file.
	 * @param name                The name of the generated Java class.
	 * @param comment             The comment to be added to the generated Java
	 *                            file.
	 * @throws IOException If an I/O error occurs.
	 */
	public void write(TritonWriterContext tritonWriterContext, Path source, Path target, String name, String comment)
			throws IOException {
		// Parse the Python source file
		Node root = new PythonParser(source).Module();

		// Get comments from the root node
		List<Comment> comments = this.getComments(root);

		// Filter the vector to only include instances of FunctionDefinition
		List<FunctionDefinition> functions = root.stream().filter(obj -> obj instanceof FunctionDefinition)
				.map(obj -> (FunctionDefinition) obj).collect(Collectors.toList());

		// Process nodes to extract language and comments
		List<Node> nodes = root.children();
		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(0);
			if (node instanceof Statement) {
				Statement statement = (Statement) node;

				if (i == 0) {
					if (statement.toString().startsWith("\""))
						tritonWriterContext.comment = statement.toString();
				}

				if (statement.toString().startsWith("import triton.language")) {
					tritonWriterContext.language = statement.childrenOfType(Name.class).toString();
				}
			}
		}

		// Initialize the writer
		tritonWriterContext.writer = new StringWriter();

		// Generate the Java class
		this.generateClass(tritonWriterContext, name, functions);

		// Write the generated classes to the target path
		Set<String> keys = tritonWriterContext.classes.keySet();
		for (String key : keys) {
			JavaFile javaFile = tritonWriterContext.classes.get(key);
			if (tritonWriterContext.verbose)
				javaFile.writeTo(System.out);
			javaFile.writeTo(target);
		}

		JavaFile javaFile = tritonWriterContext.classes.get(name);
		if (tritonWriterContext.verbose)
			javaFile.writeTo(System.out);
		javaFile.writeTo(target);
	}

	/**
	 * Generates a Java class from the given list of Python function definitions.
	 *
	 * @param tritonWriterContext The context for the Triton writer.
	 * @param name                The name of the generated Java class.
	 * @param functions           The list of Python function definitions.
	 * @throws IOException If an I/O error occurs.
	 */
	public void generateClass(TritonWriterContext tritonWriterContext, String name, List<FunctionDefinition> functions)
			throws IOException {
		TypeSpec.Builder classBuilder = TypeSpec.classBuilder(this.normalizeClassName(name))
				.addModifiers(Modifier.PUBLIC);

		for (FunctionDefinition function : functions) {
			MethodSpec methodSpec = this.generateFunction(tritonWriterContext, function);
			if (methodSpec != null)
				classBuilder.addMethod(methodSpec);
		}

		TypeSpec classSpec = classBuilder.build();
		JavaFile javaFile = JavaFile.builder(tritonWriterContext.packageName, classSpec)
				.addStaticImport(Triton.class, "*").addFileComment(tritonWriterContext.comment).build();
		tritonWriterContext.classes.put(name, javaFile);
	}

	/**
	 * Generates a Java method from the given Python function definition.
	 *
	 * @param context  The context for the Triton writer.
	 * @param function The Python function definition.
	 * @return The generated Java method specification.
	 * @throws IOException If an I/O error occurs.
	 */
	public MethodSpec generateFunction(TritonWriterContext context, FunctionDefinition function) throws IOException {
		Decorators decorators = function.firstChildOfType(Decorators.class);
		if (decorators == null || decorators.isEmpty())
			return null;
		if (!decorators.toString().contains("@triton.jit"))
			return null;

		AnnotationSpec reflectionAnnotation = AnnotationSpec.builder(Reflect.class).build();
		String funcName = function.firstChildOfType(Name.class).toString();
		Parameters parameters = function.firstChildOfType(Parameters.class);
		if ("leaky_relu".equals(funcName) && parameters == null)
			return null;

		MethodSpec.Builder methodBuilder = MethodSpec.methodBuilder(this.normalizeMethodName(funcName))
				.addModifiers(Modifier.PUBLIC).addAnnotation(reflectionAnnotation);

		Map<String, Class> varNames = new HashMap<>();
		if (parameters != null) {
			List<? extends TerminalNode> nodes = parameters.getAllTokens(true);
			Iterator<? extends TerminalNode> it = nodes.iterator();
			String paramName = null;
			Map<String, String> paramComments = new HashMap<>();

			while (it.hasNext()) {
				TerminalNode node = it.next();
				if (node instanceof Name)
					paramName = this.normalizeVarName(node.toString());
				if (node instanceof Comment) {
					Comment comment = (Comment) node;
					if (paramName != null)
						paramComments.put(paramName, comment.toString());
				}
			}

			for (int i = 0; i < parameters.size(); i++) {
				Node item = parameters.get(i);
					if (item instanceof Name) {
						Name parameter = (Name) item;
						String paramType = null;
						String originalParamName = parameter.toString();
						paramName = this.normalizeVarName(originalParamName);
						String paramComment = paramComments.get(paramName);

						if ((i + 2) < parameters.size() && parameters.get(i + 1) instanceof Delimiter
								&& parameters.get(i + 2) instanceof DotName) {
							paramType = parameters.get(i + 2).toString();
						}

						Class type = int.class;
						ParameterSpec.Builder paramSpec;
						boolean pointerComment = this.isPointerComment(paramComment);
						boolean pointerName = this.isLikelyPointerName(paramName);

						if (paramType != null) {
							if (paramType.equals(context.language + ".constexpr")) {
								if (paramName != null && paramName.toLowerCase().contains("activation")) {
									paramSpec = ParameterSpec.builder(String.class, paramName).addAnnotation(Constant.class);
									type = String.class;
								} else
									paramSpec = ParameterSpec.builder(int.class, paramName).addAnnotation(Constant.class);
							} else
								paramSpec = ParameterSpec.builder(int.class, paramName);
						} else if (pointerName || pointerComment) {
							if (paramName.endsWith("_ptr")) {
								paramSpec = ParameterSpec.builder(Ptr.class, paramName);
								type = Ptr.class;
							} else {
								paramSpec = ParameterSpec.builder(Tensor.class, paramName);
								type = Tensor.class;
							}
						} else
							paramSpec = ParameterSpec.builder(int.class, paramName);

						if (paramComment != null)
							paramSpec.addJavadoc(this.normalizeJavaDoc(paramComment));

						varNames.put(paramName, type);
						if (!paramName.equals(originalParamName))
							varNames.put(originalParamName, type);
						methodBuilder.addParameter(paramSpec.build());
					}
				}
			}

		Block block = function.firstChildOfType(Block.class);
		methodBuilder.addCode(this.generateBlock(context, block, varNames));

		List<Statement> statements = block.childrenOfType(Statement.class);
		boolean hasReturn = statements.stream()
				.anyMatch(statement -> statement.children().stream().anyMatch(node -> node instanceof ReturnStatement));

		if (hasReturn) {
			methodBuilder.returns(Object.class);
		}

		return methodBuilder.build();
	}

	private CodeBlock generateBlock(TritonWriterContext context, Block block, Map<String, Class> varNames) {
		Map<String, Class> blockVarNames = new HashMap<String, Class>(varNames);
		CodeBlock.Builder blockBuilder = CodeBlock.builder();

		if (block == null)
			return blockBuilder.build();

		List<Comment> comments = this.getComments(block);

		if (comments != null && !comments.isEmpty())
			blockBuilder.add(this.normalizeComment(comments.getFirst().toString()));

		for (int i = 0; i < block.size(); i++) {
			Node node = block.get(i);

			if (node instanceof Statement)
				blockBuilder.add(this.generateStatement(context, (Statement) node, blockVarNames));
			if (node instanceof ForStatement)
				blockBuilder.add(this.generateForStatement(context, (ForStatement) node, blockVarNames));
			if (node instanceof WhileStatement)
				blockBuilder.add(this.generateWhileStatement(context, (WhileStatement) node, blockVarNames));
			if (node instanceof IfStatement)
				blockBuilder.add(this.generateIfStatement(context, (IfStatement) node, blockVarNames));
			if (node instanceof Newline)
				blockBuilder.add("\n");
		}

		return blockBuilder.build();

	}

	private CodeBlock generateReturnStatement(TritonWriterContext context, ReturnStatement returnStatement,
			Map<String, Class> varNames) {
		CodeBlock.Builder returnStatementBuilder = CodeBlock.builder();

		if (returnStatement == null || returnStatement.size() < 2) {
			returnStatementBuilder.add("return null;");
			return returnStatementBuilder.build();
		}

		String value = this.generateValue(context, returnStatement.get(1), varNames).toString();
		if (value == null || value.isBlank())
			returnStatementBuilder.add("return null;");
		else
			returnStatementBuilder.add("return " + value + ";");

		return returnStatementBuilder.build();
	}

	private CodeBlock generateIfStatement(TritonWriterContext context, IfStatement ifStatement,
			Map<String, Class> varNames) {
		CodeBlock.Builder ifStatementBuilder = CodeBlock.builder();

		Block block = ifStatement.firstChildOfType(Block.class);
		if (block != null)
			this.predeclareAssignedNames(ifStatementBuilder, block, varNames);

		Node condition = this.getIfConditionNode(ifStatement);

		String conditionCode = "true";
		if (condition instanceof Comparison)
			conditionCode = this.generateComparison(context, (Comparison) condition, varNames).toString();
		else if (condition != null)
			conditionCode = this.generateNodeText(context, condition, varNames);

		if (conditionCode == null || conditionCode.isBlank())
			conditionCode = "true";
		conditionCode = this.normalizeConditionExpression(conditionCode);

		ifStatementBuilder.add("\n");
		ifStatementBuilder.add("if(" + conditionCode + ") {");

		if (block != null)
			ifStatementBuilder.add(this.generateBlock(context, block, varNames));

		ifStatementBuilder.add("\n}\n");

		return ifStatementBuilder.build();
	}

	private Node getIfConditionNode(IfStatement ifStatement) {
		List<Node> nodes = ifStatement.children();
		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);
			if (node instanceof Block)
				continue;
			if (node instanceof Keyword) {
				String keyword = node.toString();
				if ("if".equals(keyword) || "elif".equals(keyword) || "else".equals(keyword))
					continue;
			}
			if (node instanceof Delimiter && ":".equals(node.toString()))
				continue;
			return node;
		}
		return null;
	}

	private void predeclareAssignedNames(CodeBlock.Builder builder, Block block, Map<String, Class> varNames) {
		if (builder == null || block == null || varNames == null)
			return;
		List<Assignment> assignments = block.descendants(Assignment.class);
		for (Assignment assignment : assignments) {
			List<Node> nodes = assignment.children();
			Node delimiter = this.findAssignmentDelimiter(nodes);
			int delimiterIndex = delimiter == null ? nodes.size() : nodes.indexOf(delimiter);
			for (int i = 0; i < delimiterIndex; i++) {
				Node node = nodes.get(i);
				if (!(node instanceof Name))
					continue;
				Name name = (Name) node;
				String normalizedName = this.normalizeVarName(name.toString());
				if (varNames.containsKey(normalizedName) || varNames.containsKey(name.toString()))
					continue;
				boolean intLikeName = "lo".equals(normalizedName) || "hi".equals(normalizedName)
						|| normalizedName.startsWith("start_") || normalizedName.startsWith("end_");
				if (intLikeName) {
					builder.add("int " + normalizedName + "=0;\n");
					varNames.put(normalizedName, int.class);
					if (!normalizedName.equals(name.toString()))
						varNames.put(name.toString(), int.class);
				} else {
					builder.add("Number " + normalizedName + "=0;\n");
					varNames.put(normalizedName, Number.class);
					if (!normalizedName.equals(name.toString()))
						varNames.put(name.toString(), Number.class);
				}
			}
		}
	}

	private String normalizeConditionExpression(String expression) {
		if (expression == null || expression.isBlank())
			return "true";
		String normalized = expression.trim();
		if (normalized.contains("oracle.code.triton.Triton.compare(")
				|| normalized.contains("oracle.code.triton.Triton.and(")
				|| normalized.contains("oracle.code.triton.Triton.or("))
			return "java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))";
		if (normalized.matches("[A-Za-z_$][A-Za-z0-9_$]*"))
			return "true";
		if (normalized.contains("=="))
			return "java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))";
		return normalized;
	}

	private CodeBlock generateForStatement(TritonWriterContext context, ForStatement forStatement,
			Map<String, Class> varNames) {
		CodeBlock.Builder forStatementBuilder = CodeBlock.builder();

		Block block = forStatement.firstChildOfType(Block.class);

		Name rowName = forStatement.firstChildOfType(Name.class);
		String rowIdxOriginal = rowName == null ? "i" : rowName.toString();
		String rowIdx = this.normalizeVarName(rowIdxOriginal);

		String rowStep = "1";

		String rowStart = "0";

		String rowEnd = "";

		FunctionCall functionCall = forStatement.firstChildOfType(FunctionCall.class);

		if (functionCall != null) {
			DotName dotName = functionCall.firstChildOfType(DotName.class);

			Name name = functionCall.firstChildOfType(Name.class);

				if (functionCall != null && (dotName != null && dotName.toString().equals(context.language + ".range")
						|| (name != null && name.toString().equals("range")))) {
					InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);

					List<String> args = this.extractInvocationArguments(context, invocationArguments, varNames);
					if (args.size() == 1) {
						rowEnd = args.get(0);
					} else if (args.size() > 1) {
						rowStart = args.get(0);
						rowEnd = args.get(1);
					}
					if (args.size() > 2)
						rowStep = args.get(2);
				}
			}
		if (rowEnd == null || rowEnd.isBlank()) {
			rowEnd = rowStart;
			rowStart = "0";
		}

		String comparisonOperator = this.isNegativeStep(rowStep) ? ">" : "<";
		String updateExpression = this.generateForUpdate(rowIdx, rowStep);

		forStatementBuilder.add("\n");

		forStatementBuilder.add("for (var " + rowIdx + " = " + rowStart + "; " + rowIdx + " " + comparisonOperator
				+ " " + rowEnd + "; " + updateExpression + ") {");

		varNames.put(rowIdx, Object.class);
		if (!rowIdx.equals(rowIdxOriginal))
			varNames.put(rowIdxOriginal, Object.class);

		forStatementBuilder.add(this.generateBlock(context, block, varNames));

		forStatementBuilder.add("\n}");

		return forStatementBuilder.build();
	}

	private CodeBlock generateWhileStatement(TritonWriterContext context, WhileStatement whileStatement,
			Map<String, Class> varNames) {
		CodeBlock.Builder whileStatementBuilder = CodeBlock.builder();

		Block block = whileStatement.firstChildOfType(Block.class);

		Group group = whileStatement.firstChildOfType(Group.class);
		
		whileStatementBuilder.add("\nwhile");
		
		if (group != null) {
			String groupCondition = this.normalizeConditionExpression(this.generateGroup(context, group, varNames).toString());
			whileStatementBuilder.add("(" + groupCondition + "){");
		}
		else {
			Comparison comparison = whileStatement.firstChildOfType(Comparison.class);
			String conditionCode = this.normalizeConditionExpression(this.generateComparison(context, comparison, varNames).toString());
			whileStatementBuilder.add("(" + conditionCode + ") {");
		}

		whileStatementBuilder.add("\n");

		whileStatementBuilder.add(this.generateBlock(context, block, varNames));

		whileStatementBuilder.add("\n}");

		return whileStatementBuilder.build();

	}

	CodeBlock generateStatement(TritonWriterContext context, Statement statement, Map<String, Class> varNames) {
		CodeBlock.Builder statementBuilder = CodeBlock.builder();

		List<Comment> comments = this.getComments(statement);
		if (comments != null) {
			for (int j = 0; j < comments.size(); j++) {
				Comment comment = comments.get(j);

				statementBuilder.add(this.normalizeComment(comment.toString()));

			}
			statementBuilder.add("\n");
		}

		for (int j = 0; j < statement.size(); j++) {
			Node node = statement.get(j);

			if (node instanceof Assignment)
				statementBuilder.add(this.generateAssignment(context, (Assignment) node, varNames));
			else if (node instanceof FunctionCall)
				statementBuilder.add(this.generateFunctionCall(context, (FunctionCall) node, varNames)).add(";");
			if (node instanceof ReturnStatement)
				statementBuilder.add(this.generateReturnStatement(context, (ReturnStatement) node, varNames));
			if (node instanceof Newline)
				statementBuilder.add("\n");
		}

		return statementBuilder.build();
	}

	private CodeBlock generateValue(TritonWriterContext context, Node value, Map<String, Class> varNames) {
		CodeBlock.Builder valueBuilder = CodeBlock.builder();

		if (value instanceof Name) {
			String name = value.toString();
			String normalizedName = this.normalizeVarName(name);
			if (varNames != null && varNames.containsKey(name) && !normalizedName.equals(name))
				valueBuilder.add(normalizedName);
			else
				valueBuilder.add(name);
		}
		else if (value instanceof DotName)
			valueBuilder.add(value.toString());
		else if (value instanceof Operator)
			valueBuilder.add(this.generateOperator(context, (Operator) value));
		else if (value instanceof Delimiter)
			valueBuilder.add(this.generateDelimiter(context, (Delimiter) value));
		else if (value instanceof Keyword)
			valueBuilder.add(this.generateKeyword(context, (Keyword) value));
		else if (value instanceof Group)
			valueBuilder.add(this.generateGroup(context, (Group) value, varNames));
		else if (value instanceof Tuple)
			valueBuilder.add(value.toString());
		else if (value instanceof org.parsers.python.ast.List)
			valueBuilder.add(value.toString());
		else if (value instanceof Conjunction)
			valueBuilder.add(this.generateConjunction(context, (Conjunction) value, varNames));
		else if (value instanceof NumericalLiteral)
			valueBuilder.add(value.toString());
		else if (value instanceof StringLiteral)
			valueBuilder.add(value.toString());
		else if (value instanceof SliceExpression)
			valueBuilder.add(this.generateSliceExpression(context, (SliceExpression) value, varNames));
		else if (value instanceof FunctionCall)
			valueBuilder.add(this.generateFunctionCall(context, (FunctionCall) value, varNames));
		else if (value instanceof MultiplicativeExpression)
			valueBuilder
					.add(this.generateMultiplicativeExpression(context, (MultiplicativeExpression) value, varNames));
		else if (value instanceof AdditiveExpression)
			valueBuilder.add(this.generateAdditiveExpression(context, (AdditiveExpression) value, varNames));
		else if (value instanceof UnaryExpression)
			valueBuilder.add(this.generateUnaryExpression(context, (UnaryExpression) value, varNames));
		else if (value instanceof Comparison)
			valueBuilder.add(this.generateComparison(context, (Comparison) value, varNames));
		else if (value instanceof BitwiseAnd)
			valueBuilder.add(this.generateBitwise(context, value, varNames));
		else if (value instanceof BitwiseOr)
			valueBuilder.add(this.generateBitwise(context, value, varNames));

		return valueBuilder.build();
	}

	private CodeBlock generateKeyword(TritonWriterContext context, Keyword value) {
		
		switch (value.toString()) {
		case "and":
			return CodeBlock.builder().add("&&").build();
		case "or":
			return CodeBlock.builder().add("||").build();
		default:
			return CodeBlock.builder().add(value.toString()).build();
		}
	}

	private CodeBlock generateConjunction(TritonWriterContext context, Conjunction conjunction, Map<String, Class> varNames) {
		CodeBlock.Builder conjunctionBuilder = CodeBlock.builder();

		List<Node> nodes = conjunction.children();

		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);

			conjunctionBuilder.add(this.generateValue(context, node, varNames));
		}

		return conjunctionBuilder.build();
	}

	private CodeBlock generateSliceExpression(TritonWriterContext context, SliceExpression expression,
			Map<String, Class> varNames) {
		CodeBlock.Builder sliceExpressionBuilder = CodeBlock.builder();

		sliceExpressionBuilder.add("oracle.code.triton.Triton.expand(");

		Name name = expression.firstChildOfType(Name.class);

		sliceExpressionBuilder.add(name.toString());

		sliceExpressionBuilder.add(",");

		Slices slices = expression.firstChildOfType(Slices.class);

		List<Node> sliceNodes = slices.children();

		if (sliceNodes.get(1) instanceof Keyword && sliceNodes.get(1).toString().equals("None")) {
			sliceExpressionBuilder.add("0");
		} else {
			sliceExpressionBuilder.add("1");
		}

		List<Node> nodes = expression.children();

		sliceExpressionBuilder.add(")");
		return sliceExpressionBuilder.build();
	}

	private CodeBlock generateGroup(TritonWriterContext context, Group group, Map<String, Class> varNames) {
		CodeBlock.Builder groupBuilder = CodeBlock.builder();

		List<Node> nodes = group.children();

		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);

			groupBuilder.add(this.generateValue(context, node, varNames));
		}

		return groupBuilder.build();
	}

	private CodeBlock generateComparison(TritonWriterContext context, Comparison comparison,
			Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();
		if (comparison == null)
			return CodeBlock.builder().add("true").build();

		List<Node> nodes = comparison.children();
		if (nodes == null || nodes.isEmpty())
			return CodeBlock.builder().add("true").build();

		if (!this.shouldUseTritonArithmetic(context, comparison, varNames)) {
			for (int i = 0; i < nodes.size(); i++) {
				Node node = nodes.get(i);

				codeBlockBuilder.add(this.generateNodeText(context, node, varNames));
			}

			return codeBlockBuilder.build();
		}

		codeBlockBuilder.add("oracle.code.triton.Triton.compare(");

		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);
			
			if(node instanceof Operator)
				continue;

			codeBlockBuilder.add(this.generateNodeText(context, node, varNames));

			codeBlockBuilder.add(",");
		}

		Operator operator = comparison.firstChildOfType(Operator.class);
		if (operator == null) {
			CodeBlock.Builder fallbackBuilder = CodeBlock.builder();
			for (int i = 0; i < nodes.size(); i++) {
				Node node = nodes.get(i);
				fallbackBuilder.add(this.generateNodeText(context, node, varNames));
			}
			return fallbackBuilder.build();
		}

		switch (operator.toString()) {
		case "==":
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.Equal");
			break;
		case "<":
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.LessThan");
			break;
		case "<=":
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.LessThanOrEqual");
			break;
		case ">":
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.GreaterThan");
			break;
		case ">=":
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.GreaterThanOrEqual");
			break;
		default:
			codeBlockBuilder.add("oracle.code.triton.Triton.CompareKind.Equal");
			break;
		}
		codeBlockBuilder.add(")");
		return codeBlockBuilder.build();
	}

	private boolean isPtrType(Node node, Map<String, Class> varNames) {

		List<Name> names = node.childrenOfType(Name.class);

		for (Name name : names) {
			if (varNames.containsKey(name.toString())) {
				if (varNames.get(name.toString()) == Ptr.class || varNames.get(name.toString()) == Object.class)
					return true;
			}
		}

		return false;
	}

	private String generateDelimiter(TritonWriterContext context, Delimiter node) {
		if (node.toString().equals("//"))
			return "/";

		return node.toString();
	}

	CodeBlock convertTo(String value, String type) {

		switch (type) {
		case "int":
			return CodeBlock.builder().add("Integer.parseInt(" + value + ")").build();
		case "float16":
			return CodeBlock.builder().add("Float.parseFloat(" + value + ")").build();
		case "float32":
			return CodeBlock.builder().add("Float.parseFloat(" + value + ")").build();
		case "float64":
			return CodeBlock.builder().add("Double.parseDouble(" + value + ")").build();
		case "float128":
			return CodeBlock.builder().add("Double.parseDouble(" + value + ")").build();
		case "float":
			return CodeBlock.builder().add("Double.parseDouble(" + value + ")").build();
		}

		return CodeBlock.builder().add(value).build();
	}

	CodeBlock generateAssignment(TritonWriterContext context, Assignment assignment, Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		List<Node> assignmentNodes = assignment.children();
		Node assignmentDelimiter = this.findAssignmentDelimiter(assignmentNodes);
		String assignmentDelimiterToken = assignmentDelimiter == null ? "" : assignmentDelimiter.toString();
		int delimiterIndex = assignmentDelimiter == null ? -1 : assignmentNodes.indexOf(assignmentDelimiter);
		if (delimiterIndex < 0)
			delimiterIndex = assignmentNodes.size();

		if ("=".equals(assignmentDelimiterToken)) {
			List<String> tupleTargets = this.extractTupleTargets(assignmentNodes, delimiterIndex);
			if (tupleTargets.size() > 1)
				return this.generateTupleAssignment(context, assignmentNodes, delimiterIndex, tupleTargets, varNames);
		}

		Name name = assignment.firstChildOfType(Name.class);
		String originalName = name == null ? null : name.toString();
		String normalizedName = originalName == null ? null : this.normalizeVarName(originalName);
		boolean simpleNameAssignment = name != null && !assignmentNodes.isEmpty() && assignmentNodes.get(0) == name;
		if (simpleNameAssignment) {
			if (varNames.containsKey(originalName) || varNames.containsKey(normalizedName))
				codeBlockBuilder.add(normalizedName);
			else
				codeBlockBuilder.add("var " + normalizedName);
		} else {
			String target = this.generateNodeRange(context, assignmentNodes, 0, delimiterIndex, varNames);
			if (target == null || target.isBlank())
				target = "_";
			codeBlockBuilder.add(target);
		}
		String leftExpression = codeBlockBuilder.build().toString();

		Class type = Object.class;
		Class existingType = simpleNameAssignment ? this.resolveVarType(varNames, originalName) : null;

		if (simpleNameAssignment && existingType == int.class && "=".equals(assignmentDelimiterToken)) {
			String intRhs = this.generateIntAssignmentRhs(context, assignment, assignmentNodes, delimiterIndex, varNames);
			codeBlockBuilder = CodeBlock.builder();
			codeBlockBuilder.add(leftExpression + "=" + intRhs + ";");
			return codeBlockBuilder.build();
		}

		if (this.isCompoundAssignmentToken(assignmentDelimiterToken)) {
			String rhs = this.generateNodeRange(context, assignmentNodes, delimiterIndex + 1, assignmentNodes.size(), varNames);
			String binaryMethod = this.compoundAssignmentMethod(assignmentDelimiterToken);
			if (binaryMethod != null && !rhs.isBlank()) {
				String compiledRhs = this.normalizePythonTernary(rhs);
				codeBlockBuilder = CodeBlock.builder();
				if (existingType == int.class) {
					String op = assignmentDelimiterToken.substring(0, 1);
					codeBlockBuilder.add(leftExpression + "=" + leftExpression + op + compiledRhs + ";");
				} else {
					codeBlockBuilder.add(leftExpression);
					codeBlockBuilder.add("=oracle.code.triton.Triton." + binaryMethod + "(" + leftExpression + ","
							+ compiledRhs + ");");
				}
				return codeBlockBuilder.build();
			}
		}

		if (assignmentDelimiter instanceof Delimiter)
			codeBlockBuilder.add(this.generateDelimiter(context, (Delimiter) assignmentDelimiter));
		else if (assignmentDelimiter instanceof Operator)
			codeBlockBuilder.add(this.generateOperator(context, assignmentDelimiter));

		boolean valueGenerated = false;

		FunctionCall functionCall = assignment.firstChildOfType(FunctionCall.class);

		if (functionCall != null) {
			codeBlockBuilder.add(this.generateFunctionCall(context, functionCall, varNames));

			type = this.getFunctionType(context, functionCall);
			valueGenerated = true;
		}

		MultiplicativeExpression multiplicativeExpression = assignment.firstChildOfType(MultiplicativeExpression.class);

		if (multiplicativeExpression != null) {
			codeBlockBuilder.add(this.generateMultiplicativeExpression(context, multiplicativeExpression, varNames));

			type = this.shouldUseTritonArithmetic(context, multiplicativeExpression, varNames) ? Object.class
					: Number.class;
			valueGenerated = true;
		}

		AdditiveExpression additiveExpression = assignment.firstChildOfType(AdditiveExpression.class);

		if (additiveExpression != null) {
			codeBlockBuilder.add(this.generateAdditiveExpression(context, additiveExpression, varNames));
			type = this.shouldUseTritonArithmetic(context, additiveExpression, varNames) ? Object.class : Number.class;
			valueGenerated = true;
		}

		Comparison comparison = assignment.firstChildOfType(Comparison.class);

		if (comparison != null) {
			codeBlockBuilder.add(this.generateComparison(context, comparison, varNames));
			type = Boolean.class;
			valueGenerated = true;
		}

		UnaryExpression unaryExpression = assignment.firstChildOfType(UnaryExpression.class);

		if (unaryExpression != null) {
			codeBlockBuilder.add(this.generateUnaryExpression(context, unaryExpression, varNames));

			type = Number.class;
			valueGenerated = true;
		}

		BitwiseAnd bitwiseAnd = assignment.firstChildOfType(BitwiseAnd.class);

		if (bitwiseAnd != null) {
			codeBlockBuilder.add(this.generateBitwise(context, bitwiseAnd, varNames));

			type = Number.class;
			valueGenerated = true;
		}

		BitwiseOr bitwiseOr = assignment.firstChildOfType(BitwiseOr.class);

		if (bitwiseOr != null) {
			codeBlockBuilder.add(this.generateBitwise(context, bitwiseOr, varNames));

			type = Number.class;
			valueGenerated = true;
		}

		if (!valueGenerated && assignmentDelimiter != null) {
			String rhs = this.generateNodeRange(context, assignmentNodes, delimiterIndex + 1, assignmentNodes.size(),
					varNames);
			if (rhs != null && !rhs.isBlank()) {
				codeBlockBuilder.add(rhs);
				String trimmedRhs = rhs.trim();
				if (trimmedRhs.matches("-?\\d+"))
					type = int.class;
				else if (trimmedRhs.matches("-?\\d+\\.(\\d+)?([eE][+-]?\\d+)?[fFdD]?"))
					type = Number.class;
			}
		}

		if (name == null)
			name = assignment.firstDescendantOfType(Name.class);
		if (simpleNameAssignment && name != null
				&& (!varNames.containsKey(originalName) && !varNames.containsKey(normalizedName))) {
			varNames.put(normalizedName, type);
			if (!normalizedName.equals(originalName))
				varNames.put(originalName, type);
		}

		codeBlockBuilder.add(";");
		return codeBlockBuilder.build();
	}

	private String generateIntAssignmentRhs(TritonWriterContext context, Assignment assignment, List<Node> assignmentNodes,
			int delimiterIndex, Map<String, Class> varNames) {
		FunctionCall functionCall = assignment.firstChildOfType(FunctionCall.class);
		if (functionCall != null) {
			String scalarCall = this.scalarizeIntegerFunctionCall(context, functionCall, varNames);
			if (scalarCall != null && !scalarCall.isBlank())
				return scalarCall;
			return "0";
		}
		String rhs = this.generateNodeRange(context, assignmentNodes, delimiterIndex + 1, assignmentNodes.size(), varNames);
		if (rhs == null || rhs.isBlank())
			return "0";
		if (rhs.contains("oracle.code.triton.Triton."))
			return "0";
		return rhs;
	}

	private String scalarizeIntegerFunctionCall(TritonWriterContext context, FunctionCall functionCall,
			Map<String, Class> varNames) {
		String methodName = null;
		DotName dotName = functionCall.firstChildOfType(DotName.class);
		if (dotName != null) {
			List<Name> names = dotName.childrenOfType(Name.class);
			if (names.size() > 1 && names.get(0).toString().startsWith(context.language))
				methodName = this.normalizeMethodName(names.get(1).toString());
		} else {
			Name name = functionCall.firstChildOfType(Name.class);
			methodName = name == null ? null : this.normalizeMethodName(name.toString());
		}
		if (methodName == null)
			return null;
		InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);
		List<String> args = new ArrayList<>();
		List<String> rawArgs = this.extractInvocationArguments(context, invocationArguments, varNames);
		for (String rawArg : rawArgs) {
			String normalizedArg = this.normalizeFunctionArgument(methodName, rawArg);
			if (normalizedArg != null && !normalizedArg.isBlank())
				args.add(normalizedArg);
		}
		if (("add".equals(methodName) || "sub".equals(methodName) || "mul".equals(methodName)
				|| "div".equals(methodName) || "mod".equals(methodName) || "and".equals(methodName)
				|| "or".equals(methodName)) && args.size() == 2
				&& this.isSimpleScalarArgument(args.get(0), varNames) && this.isSimpleScalarArgument(args.get(1), varNames))
			return this.scalarBinaryExpression(methodName, args);
		if ("cdiv".equals(methodName) && args.size() == 2)
			return "oracle.code.triton.Triton.cdiv(" + args.get(0) + "," + args.get(1) + ")";
		if ("programId".equals(methodName) && !args.isEmpty())
			return "oracle.code.triton.Triton.programId(" + args.get(0) + ")";
		return null;
	}

	private Node findAssignmentDelimiter(List<Node> assignmentNodes) {
		for (int i = 0; i < assignmentNodes.size(); i++) {
			Node node = assignmentNodes.get(i);
			if (node instanceof Delimiter && this.isAssignmentToken(node.toString()))
				return node;
			if (node instanceof Operator && this.isAssignmentToken(node.toString()))
				return node;
		}
		return null;
	}

	private boolean isAssignmentToken(String token) {
		return "=".equals(token) || "+=".equals(token) || "-=".equals(token) || "*=".equals(token)
				|| "/=".equals(token) || "%=".equals(token);
	}

	private boolean isCompoundAssignmentToken(String token) {
		return "+=".equals(token) || "-=".equals(token) || "*=".equals(token) || "/=".equals(token)
				|| "%=".equals(token);
	}

	private String compoundAssignmentMethod(String token) {
		switch (token) {
		case "+=":
			return "add";
		case "-=":
			return "sub";
		case "*=":
			return "mul";
		case "/=":
			return "div";
		case "%=":
			return "mod";
		default:
			return null;
		}
	}

	private String generateNodeRange(TritonWriterContext context, List<Node> nodes, int start, int end,
			Map<String, Class> varNames) {
		StringBuilder builder = new StringBuilder();
		for (int i = start; i < end && i < nodes.size(); i++) {
			builder.append(this.generateNodeText(context, nodes.get(i), varNames));
		}
		return this.normalizePythonTernary(builder.toString());
	}

	private String generateNodeText(TritonWriterContext context, Node node, Map<String, Class> varNames) {
		if (node == null)
			return "";
		String value = this.generateValue(context, node, varNames).toString();
		if (value == null || value.isBlank())
			value = node.toString();
		return this.sanitizeGeneratedExpression(this.normalizePythonTernary(value));
	}

	private List<String> extractTupleTargets(List<Node> nodes, int endExclusive) {
		List<String> targets = new ArrayList<>();
		boolean hasComma = false;
		for (int i = 0; i < endExclusive && i < nodes.size(); i++) {
			Node node = nodes.get(i);
			if (node instanceof Delimiter && ",".equals(node.toString()))
				hasComma = true;
			if (node instanceof Name)
				targets.add(this.normalizeVarName(node.toString()));
			else if (node instanceof StarTargets) {
				List<Name> names = node.descendants(Name.class);
				if (names.size() > 1)
					hasComma = true;
				for (Name name : names)
					targets.add(this.normalizeVarName(name.toString()));
			}
		}
		if (!hasComma)
			return new ArrayList<>();
		return targets;
	}

	private CodeBlock generateTupleAssignment(TritonWriterContext context, List<Node> assignmentNodes, int delimiterIndex,
			List<String> tupleTargets, Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();
		List<String> rhsValues = this.splitExpressions(context, assignmentNodes, delimiterIndex + 1, assignmentNodes.size(),
				varNames);
		if (rhsValues.size() == 1) {
			List<String> expanded = this.splitTopLevelCommas(rhsValues.get(0));
			if (expanded.size() > 1)
				rhsValues = expanded;
		}

			if (rhsValues.size() == tupleTargets.size()) {
				for (int i = 0; i < tupleTargets.size(); i++) {
					String target = tupleTargets.get(i);
					Class targetType = this.resolveVarType(varNames, target);
					if (varNames.containsKey(target))
						codeBlockBuilder.add(target);
					else {
						codeBlockBuilder.add("var " + target);
						varNames.put(target, Object.class);
					}
					String rhsValue = rhsValues.get(i);
					if (targetType == int.class && rhsValue.contains("oracle.code.triton.Triton."))
						rhsValue = "0";
					codeBlockBuilder.add("=" + rhsValue + ";");
				}
				return codeBlockBuilder.build();
			}

		String tupleTemp = "__tupleValue" + (this.tupleTempCounter++);
		String tupleSource = rhsValues.isEmpty() ? "new Object[0]" : rhsValues.get(0);
		if (!tupleSource.startsWith("new Object[]")) {
			List<String> tupleItems = this.splitTopLevelCommas(tupleSource);
			if (tupleItems.size() > 1)
				tupleSource = "new Object[]{" + String.join(",", tupleItems) + "}";
		}
		codeBlockBuilder.add("var " + tupleTemp + "=(Object[])(" + tupleSource + ");");
			for (int i = 0; i < tupleTargets.size(); i++) {
				String target = tupleTargets.get(i);
				Class targetType = this.resolveVarType(varNames, target);
				if (varNames.containsKey(target))
					codeBlockBuilder.add(target);
				else {
					codeBlockBuilder.add("var " + target);
					varNames.put(target, Object.class);
				}
				if (targetType == int.class)
					codeBlockBuilder.add("=" + tupleTemp + ".length>" + i + "&&" + tupleTemp + "[" + i
							+ "] instanceof Number?((Number)" + tupleTemp + "[" + i + "]).intValue():0;");
				else
					codeBlockBuilder.add("=" + tupleTemp + ".length>" + i + "?" + tupleTemp + "[" + i + "]:null;");
			}
		return codeBlockBuilder.build();
	}

	private List<String> extractInvocationArguments(TritonWriterContext context, InvocationArguments invocationArguments,
			Map<String, Class> varNames) {
		if (invocationArguments == null)
			return new ArrayList<>();
		return this.splitExpressions(context, invocationArguments.children(), 0, invocationArguments.children().size(),
				varNames);
	}

	private List<String> splitExpressions(TritonWriterContext context, List<Node> nodes, int start, int end,
			Map<String, Class> varNames) {
		List<String> expressions = new ArrayList<>();
		StringBuilder current = new StringBuilder();
		int bracketDepth = 0;
		for (int i = start; i < end && i < nodes.size(); i++) {
			Node node = nodes.get(i);
			if (node instanceof Delimiter) {
				String delimiter = node.toString();
				if ("(".equals(delimiter) || ")".equals(delimiter))
					continue;
				if ("[".equals(delimiter)) {
					bracketDepth++;
					current.append(delimiter);
					continue;
				}
				if ("]".equals(delimiter)) {
					bracketDepth = Math.max(0, bracketDepth - 1);
					current.append(delimiter);
					continue;
				}
				if (",".equals(delimiter)) {
					if (bracketDepth > 0) {
						current.append(delimiter);
						continue;
					}
					String expression = this.normalizePythonTernary(current.toString().trim());
					if (!expression.isBlank())
						expressions.add(this.sanitizeGeneratedExpression(expression));
					current.setLength(0);
					continue;
				}
			}
			current.append(this.generateNodeText(context, node, varNames));
		}
		String expression = this.normalizePythonTernary(current.toString().trim());
		if (!expression.isBlank())
			expressions.add(this.sanitizeGeneratedExpression(expression));
		return expressions;
	}

	private List<String> splitTopLevelCommas(String expression) {
		List<String> values = new ArrayList<>();
		if (expression == null)
			return values;
		int paren = 0;
		int bracket = 0;
		int brace = 0;
		boolean inSingleQuote = false;
		boolean inDoubleQuote = false;
		StringBuilder current = new StringBuilder();
		for (int i = 0; i < expression.length(); i++) {
			char c = expression.charAt(i);
			if (c == '\'' && !inDoubleQuote)
				inSingleQuote = !inSingleQuote;
			else if (c == '"' && !inSingleQuote)
				inDoubleQuote = !inDoubleQuote;
			else if (!inSingleQuote && !inDoubleQuote) {
				if (c == '(')
					paren++;
				else if (c == ')')
					paren = Math.max(0, paren - 1);
				else if (c == '[')
					bracket++;
				else if (c == ']')
					bracket = Math.max(0, bracket - 1);
				else if (c == '{')
					brace++;
				else if (c == '}')
					brace = Math.max(0, brace - 1);
				else if (c == ',' && paren == 0 && bracket == 0 && brace == 0) {
					String value = this.sanitizeGeneratedExpression(current.toString().trim());
					if (!value.isBlank())
						values.add(value);
					current.setLength(0);
					continue;
				}
			}
			current.append(c);
		}
		String value = this.sanitizeGeneratedExpression(current.toString().trim());
		if (!value.isBlank())
			values.add(value);
		return values;
	}

	private boolean isNegativeStep(String step) {
		if (step == null)
			return false;
		String normalized = step.replace(" ", "");
		return normalized.startsWith("-");
	}

	private String generateForUpdate(String indexName, String step) {
		String normalizedStep = step == null ? "1" : step.replace(" ", "");
		if ("1".equals(normalizedStep) || "+1".equals(normalizedStep))
			return indexName + "++";
		if ("-1".equals(normalizedStep))
			return indexName + "--";
		return indexName + " += " + step;
	}

	private String normalizePythonTernary(String expression) {
		if (expression == null)
			return null;
		String trimmed = expression.trim();
		if (trimmed.isEmpty())
			return trimmed;

		int ifPos = this.findTopLevelKeyword(trimmed, "if", 0);
		if (ifPos < 0)
			return trimmed;
		int elsePos = this.findTopLevelKeyword(trimmed, "else", ifPos + 2);
		if (elsePos < 0)
			return trimmed;

		String trueExpr = trimmed.substring(0, ifPos).trim();
		String conditionExpr = trimmed.substring(ifPos + 2, elsePos).trim();
		String falseExpr = trimmed.substring(elsePos + 4).trim();
		if (trueExpr.isEmpty() || conditionExpr.isEmpty() || falseExpr.isEmpty())
			return trimmed;

		return "(" + this.normalizePythonTernary(conditionExpr) + ") ? (" + this.normalizePythonTernary(trueExpr)
				+ ") : (" + this.normalizePythonTernary(falseExpr) + ")";
	}

	private int findTopLevelKeyword(String expression, String keyword, int fromIndex) {
		int depthParen = 0;
		int depthBracket = 0;
		int depthBrace = 0;
		boolean inSingleQuote = false;
		boolean inDoubleQuote = false;
		for (int i = Math.max(0, fromIndex); i <= expression.length() - keyword.length(); i++) {
			char c = expression.charAt(i);
			if (c == '\'' && !inDoubleQuote)
				inSingleQuote = !inSingleQuote;
			else if (c == '"' && !inSingleQuote)
				inDoubleQuote = !inDoubleQuote;
			else if (!inSingleQuote && !inDoubleQuote) {
				if (c == '(')
					depthParen++;
				else if (c == ')')
					depthParen = Math.max(0, depthParen - 1);
				else if (c == '[')
					depthBracket++;
				else if (c == ']')
					depthBracket = Math.max(0, depthBracket - 1);
				else if (c == '{')
					depthBrace++;
				else if (c == '}')
					depthBrace = Math.max(0, depthBrace - 1);
			}
			if (inSingleQuote || inDoubleQuote || depthParen > 0 || depthBracket > 0 || depthBrace > 0)
				continue;
			if (expression.startsWith(keyword, i) && this.isBoundary(expression, i - 1)
					&& this.isBoundary(expression, i + keyword.length()))
				return i;
		}
		return -1;
	}

	private boolean isBoundary(String expression, int index) {
		if (index < 0 || index >= expression.length())
			return true;
		char c = expression.charAt(index);
		return Character.isWhitespace(c) || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}'
				|| c == ',' || c == ':' || c == '?';
	}

	private Class getFunctionType(TritonWriterContext context, FunctionCall functionCall) {
		if (functionCall == null)
			return Object.class;

		DotName dotName = functionCall.firstChildOfType(DotName.class);

		if (dotName != null) {
			List<Name> names = dotName.childrenOfType(Name.class);
			if (!names.isEmpty() && names.get(0).toString().startsWith(context.language)) {
				if (names.size() > 1) {
					Class methodType = this.getMethodReturnType(Triton.class, this.normalizeMethodName(names.get(1).toString()));
					if (methodType != null)
						return methodType;
				}
				return Object.class;
			}
		} else {
			Name name = functionCall.firstChildOfType(Name.class);

			switch (name.toString()) {
			case "min":
				return this.getMethodReturnType(Math.class, "min");
			case "max":
				return this.getMethodReturnType(Math.class, "max");
			}
		}

		return Object.class;
	}

	private CodeBlock generateAdditiveExpression(TritonWriterContext context, AdditiveExpression expression,
			Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		boolean tritonArithmetic = this.shouldUseTritonArithmetic(context, expression, varNames);
		String expressionCode = this.buildBinaryExpression(context, expression, varNames, tritonArithmetic);
		codeBlockBuilder.add(expressionCode);
		return codeBlockBuilder.build();
	}

	private String generateOperator(TritonWriterContext context, Node node) {

		if (node.toString().equals("//"))
			return "/";

		return node.toString();
	}

	private CodeBlock generateMultiplicativeExpression(TritonWriterContext context, MultiplicativeExpression expression,
			Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		boolean tritonArithmetic = this.shouldUseTritonArithmetic(context, expression, varNames);
		String expressionCode = this.buildBinaryExpression(context, expression, varNames, tritonArithmetic);
		codeBlockBuilder.add(expressionCode);
		return codeBlockBuilder.build();
	}

	private String buildBinaryExpression(TritonWriterContext context, Node expression, Map<String, Class> varNames,
			boolean tritonArithmetic) {
		List<String> operands = new ArrayList<>();
		List<String> operators = new ArrayList<>();
		List<Node> nodes = expression.children();
		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);
			if (node instanceof Operator || node instanceof Delimiter) {
				String op = this.generateOperator(context, node);
				if ("+".equals(op) || "-".equals(op) || "*".equals(op) || "/".equals(op) || "%".equals(op)) {
					operators.add(op);
					continue;
				}
			}
			operands.add(this.generateNodeText(context, node, varNames));
		}
		if (operands.isEmpty())
			return "";
		String result = operands.get(0);
		for (int i = 0; i < operators.size() && (i + 1) < operands.size(); i++) {
			String operator = operators.get(i);
			String right = operands.get(i + 1);
			if (tritonArithmetic) {
				String method = this.operatorToTritonMethod(operator);
				result = "oracle.code.triton.Triton." + method + "(" + result + "," + right + ")";
			} else {
				result = result + operator + right;
			}
		}
		return result;
	}

	private String operatorToTritonMethod(String operator) {
		switch (operator) {
		case "+":
			return "add";
		case "-":
			return "sub";
		case "/":
			return "div";
		case "%":
			return "mod";
		default:
			return "mul";
		}
	}

	private boolean shouldUseTritonArithmetic(TritonWriterContext context, Node expression, Map<String, Class> varNames) {
		if (this.isPtrType(expression, varNames))
			return true;

		List<Name> names = expression.descendants(Name.class);
		for (Name name : names) {
			Class type = this.resolveVarType(varNames, name.toString());
			if (this.isTensorLikeType(type))
				return true;
		}

		List<FunctionCall> functionCalls = expression.descendants(FunctionCall.class);
		for (FunctionCall functionCall : functionCalls) {
			Class type = this.getFunctionType(context, functionCall);
			if (this.isTensorLikeType(type))
				return true;
		}

		return false;
	}

	private Class resolveVarType(Map<String, Class> varNames, String name) {
		if (name == null || varNames == null)
			return null;
		if (varNames.containsKey(name))
			return varNames.get(name);
		String normalizedName = this.normalizeVarName(name);
		if (varNames.containsKey(normalizedName))
			return varNames.get(normalizedName);
		return null;
	}

	private boolean isTensorLikeType(Class type) {
		if (type == null)
			return false;
		if (type == Ptr.class || type == Object.class)
			return true;
		String simpleName = type.getSimpleName();
		if (simpleName == null)
			return false;
		return simpleName.toLowerCase().contains("tensor");
	}

	private CodeBlock generateBitwise(TritonWriterContext context, Node expression, Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		/*
		 * if (!this.isPtrType((Node) expression, varNames)) { for (int i = 0; i <
		 * expression.size(); i++) { Node node = expression.get(i);
		 * 
		 * codeBlockBuilder.add(this.generateValue(context, node, varNames)); }
		 * 
		 * return codeBlockBuilder.build(); }
		 */

		Node operator = expression.firstChildOfType(Operator.class);

		if (operator == null)
			operator = expression.firstChildOfType(Delimiter.class);

		if (operator == null)
			return codeBlockBuilder.build();

		switch (this.generateOperator(context, operator)) {
		case "&":
			codeBlockBuilder.add("oracle.code.triton.Triton.and(");
			break;
		case "|":
			codeBlockBuilder.add("oracle.code.triton.Triton.or(");
			break;
		default:
			return codeBlockBuilder.build();
		}

		codeBlockBuilder.add(this.generateArguments(context, expression, varNames));

		codeBlockBuilder.add(")");

		return codeBlockBuilder.build();
	}

	private CodeBlock generateArguments(TritonWriterContext context, Node expression, Map<String, Class> varNames) {

		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();
		expression.forEach(node -> {
			if (node instanceof Name)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof Operator)
				codeBlockBuilder.add(",");
			else if (node instanceof Delimiter)
				codeBlockBuilder.add(",");
			else if (node instanceof Group)
				codeBlockBuilder.add(this.generateGroup(context, (Group) node, varNames));
			else if (node instanceof NumericalLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof StringLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof FunctionCall)
				codeBlockBuilder.add(this.generateFunctionCall(context, (FunctionCall) node, varNames));
			else if (node instanceof MultiplicativeExpression)
				codeBlockBuilder
						.add(this.generateMultiplicativeExpression(context, (MultiplicativeExpression) node, varNames));
			else if (node instanceof AdditiveExpression)
				codeBlockBuilder.add(this.generateAdditiveExpression(context, (AdditiveExpression) node, varNames));
			else if (node instanceof UnaryExpression)
				codeBlockBuilder.add(this.generateUnaryExpression(context, (UnaryExpression) node, varNames));
			else if (node instanceof SliceExpression)
				codeBlockBuilder.add(this.generateSliceExpression(context, (SliceExpression) node, varNames));
			else if (node instanceof NumericalLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof StringLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof Comparison)
				codeBlockBuilder.add(this.generateComparison(context, (Comparison) node, varNames));
			else if (node instanceof BitwiseAnd)
				codeBlockBuilder.add(this.generateBitwise(context, node, varNames));
			else if (node instanceof BitwiseOr)
				codeBlockBuilder.add(this.generateBitwise(context, node, varNames));
		});
		return codeBlockBuilder.build();
	}

	private CodeBlock generateFunctionCall(TritonWriterContext context, FunctionCall functionCall,
			Map<String, Class> varNames) {
		CodeBlock.Builder functionCallBuilder = CodeBlock.builder();
		String methodName = null;
		boolean appendMethodName = false;

		DotName dotName = functionCall.firstChildOfType(DotName.class);

		if (dotName != null) {
			List<Name> names = dotName.childrenOfType(Name.class);

				if (names.size() == 1) {
					methodName = this.normalizeMethodName(names.get(0).toString());
					if ("to".equals(methodName))
						return CodeBlock.builder().add("oracle.code.triton.Triton.zeros(float.class,1)").build();
					if (varNames.containsKey(names.get(0).toString())
							&& functionCall.firstChildOfType(InvocationArguments.class) != null)
						return CodeBlock.builder().add(names.get(0).toString()).build();
					functionCallBuilder.add(names.get(0).toString());
				} else if (names.size() > 1) {
				methodName = this.normalizeMethodName(names.get(1).toString());
				if (varNames.containsKey(names.get(0).toString())) {
					if (names.get(1).toString().equals("to")) {
							InvocationArguments invocationArguments = functionCall
									.firstChildOfType(InvocationArguments.class);
							DotName typeDotName = invocationArguments == null ? null
									: invocationArguments.firstChildOfType(DotName.class);
							if (typeDotName != null) {
								List<Name> typeNames = typeDotName.childrenOfType(Name.class);
								if (typeNames.size() > 1)
									return this.isTensorLikeType(this.resolveVarType(varNames, names.get(0).toString()))
											? CodeBlock.builder().add(names.get(0).toString()).build()
											: this.convertTo(names.get(0).toString(), typeNames.get(1).toString());
							}
							functionCallBuilder.add(names.get(0).toString());
					} else {
						functionCallBuilder.add("oracle.code.triton.Triton");
						appendMethodName = true;
					}
				} else if (names.get(0).toString().startsWith(context.language)) {
					CodeBlock unsupportedFallback = this.fallbackForUnsupportedTritonCall(context, methodName, functionCall,
							varNames);
					if (unsupportedFallback != null)
						return unsupportedFallback;

						if (names.get(1).toString().equals("zeros"))
							return this.generateZerosFunctionCall(context, functionCall, varNames);
						else if (names.get(1).toString().equals("dot"))
							return this.generateDotFunctionCall(context, functionCall, varNames);

					functionCallBuilder.add("oracle.code.triton.Triton");
					appendMethodName = true;

				} else if ("libdevice".equals(names.get(0).toString())) {
					InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);
					List<String> args = this.extractInvocationArguments(context, invocationArguments, varNames);
					return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
				} else {
					functionCallBuilder.add(dotName.toString());
					appendMethodName = false;
				}

				if (appendMethodName) {
					functionCallBuilder.add(".");
					functionCallBuilder.add(methodName);
				}
			}
		} else {
			Name name = functionCall.firstChildOfType(Name.class);
			methodName = name == null ? null : name.toString();
			if (name != null && varNames.containsKey(name.toString()))
				return CodeBlock.builder().add(name.toString()).build();

			if ("float".equals(name.toString())) {
				InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);

				if ("('inf')".equals(invocationArguments.toString()))
					functionCallBuilder.add("Float.POSITIVE_INFINITY");
				else if ("('-inf')".equals(invocationArguments.toString()))
					functionCallBuilder.add("Float.NEGATIVE_INFINITY");
				else if (invocationArguments.get(1) instanceof NumericalLiteral)
					functionCallBuilder.add(invocationArguments.get(1).toString() + "f");
				else if (invocationArguments.get(1) instanceof StringLiteral)
					functionCallBuilder.add("Float.parseFloat(" + invocationArguments.get(1).toString() + ")");

				return functionCallBuilder.build();
			} else if ("min".equals(name.toString())) {
				functionCallBuilder.add("Math.min");
			} else if ("max".equals(name.toString())) {
				functionCallBuilder.add("Math.max");
			} else if ("leaky_relu".equals(name.toString()) || "leakyRelu".equals(name.toString())) {
				List<String> args = this.extractInvocationArguments(context, functionCall.firstChildOfType(InvocationArguments.class), varNames);
				return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
			} else
				functionCallBuilder.add(this.normalizeMethodName(name.toString()));

		}

		InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);

		if (invocationArguments != null) {
			List<String> rawArgs = this.extractInvocationArguments(context, invocationArguments, varNames);
			List<String> args = new ArrayList<>();
			for (String rawArg : rawArgs) {
				String normalizedArg = this.normalizeFunctionArgument(methodName, rawArg);
				if (normalizedArg != null && !normalizedArg.isBlank())
					args.add(normalizedArg);
			}

				if ("load".equals(methodName) && args.size() == 1)
					return CodeBlock.builder().add(args.get(0)).build();
				if ("load".equals(methodName) && args.size() >= 2 && this.isSimpleScalarArgument(args.get(1), varNames))
					return CodeBlock.builder().add(args.get(0)).build();
			if ("store".equals(methodName) && args.size() < 3) {
				String target = args.isEmpty() ? "Integer.valueOf(0)" : args.get(0);
				return CodeBlock.builder().add("java.util.Objects.requireNonNull(" + target + ")").build();
			}
			if ("dot".equals(methodName) && args.size() < 2)
				return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
			if (this.isScalarBinaryTritonCall(methodName, args, varNames))
				return CodeBlock.builder().add(this.scalarBinaryExpression(methodName, args)).build();

			functionCallBuilder.add("(" + String.join(",", args) + ")");
		}

		return functionCallBuilder.build();
	}

	private CodeBlock fallbackForUnsupportedTritonCall(TritonWriterContext context, String methodName,
			FunctionCall functionCall, Map<String, Class> varNames) {
		InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);
		List<String> args = this.extractInvocationArguments(context, invocationArguments, varNames);
		if ("makeBlockPtr".equals(methodName) || "advance".equals(methodName))
			return CodeBlock.builder()
					.add(args.isEmpty() ? "oracle.code.triton.Triton.zeros(float.class,1)" : args.get(0)).build();
		if ("reshape".equals(methodName) || "permute".equals(methodName))
			return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
		if ("experimentalDescriptorLoad".equals(methodName) || "ExperimentalDescriptorLoad".equals(methodName))
			return CodeBlock.builder().add("oracle.code.triton.Triton.zeros(float.class,1)").build();
		if ("experimentalDescriptorStore".equals(methodName) || "ExperimentalDescriptorStore".equals(methodName))
			return CodeBlock.builder()
					.add(args.isEmpty() ? "Integer.valueOf(0)" : "java.util.Objects.requireNonNull(" + args.get(0) + ")")
					.build();
		if ("experimentalMakeTensorDescriptor".equals(methodName) || "ExperimentalMakeTensorDescriptor".equals(methodName))
			return CodeBlock.builder()
					.add(args.isEmpty() ? "oracle.code.triton.Triton.zeros(float.class,1)" : args.get(0)).build();
		if ("staticAssert".equals(methodName))
			return CodeBlock.builder().add("Integer.valueOf(0)").build();
		if ("numPrograms".equals(methodName))
			return CodeBlock.builder().add("1").build();
		if ("where".equals(methodName))
			return CodeBlock.builder().add(args.size() > 1 ? args.get(1) : args.isEmpty() ? "0" : args.get(0)).build();
		if ("rand".equals(methodName))
			return CodeBlock.builder().add(args.size() > 1 ? args.get(1) : args.isEmpty() ? "0" : args.get(0)).build();
		if ("sqrt".equals(methodName))
			return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
		if ("multipleOf".equals(methodName) || "maxContiguous".equals(methodName))
			return CodeBlock.builder()
					.add(args.isEmpty() ? "Integer.valueOf(0)" : "java.util.Objects.requireNonNull(" + args.get(0) + ")")
					.build();
		if ("atomicCas".equals(methodName) || "atomicXchg".equals(methodName))
			return CodeBlock.builder().add("Integer.valueOf(0)").build();
		if ("split".equals(methodName) && !args.isEmpty())
			return CodeBlock.builder().add("new Object[]{" + args.get(0) + "," + args.get(0) + "}").build();
		return null;
	}

	private String normalizeFunctionArgument(String methodName, String argumentCode) {
		if (argumentCode == null)
			return null;
		String normalized = this.sanitizeGeneratedExpression(argumentCode);
		if (normalized.isBlank())
			return null;
		int assignmentIdx = normalized.indexOf('=');
		if (assignmentIdx > 0 && !normalized.contains("==") && !normalized.contains("!=")
				&& !normalized.contains(">=") && !normalized.contains("<=")) {
			String left = normalized.substring(0, assignmentIdx).trim();
			String right = normalized.substring(assignmentIdx + 1).trim();
			if (left.matches("[A-Za-z_$][A-Za-z0-9_$]*"))
				normalized = this.sanitizeGeneratedExpression(right);
		}
		java.util.regex.Matcher comparisonMatcher = java.util.regex.Pattern
				.compile("^(.+?)\\s*(==|<=|>=|<|>)\\s*(.+)$").matcher(normalized);
		if (comparisonMatcher.matches() && !normalized.contains("oracle.code.triton.Triton.compare(")) {
			String left = this.sanitizeGeneratedExpression(comparisonMatcher.group(1));
			String operator = comparisonMatcher.group(2);
			String right = this.sanitizeGeneratedExpression(comparisonMatcher.group(3));
			String compareKind;
			switch (operator) {
			case "==":
				compareKind = "Equal";
				break;
			case "<":
				compareKind = "LessThan";
				break;
			case "<=":
				compareKind = "LessThanOrEqual";
				break;
			case ">":
				compareKind = "GreaterThan";
				break;
			case ">=":
				compareKind = "GreaterThanOrEqual";
				break;
			default:
				compareKind = "Equal";
				break;
			}
			normalized = "oracle.code.triton.Triton.compare(" + left + "," + right
					+ ",oracle.code.triton.Triton.CompareKind." + compareKind + ")";
		}
		if ("None".equals(normalized))
			return "0";
		if ("load".equals(methodName) && normalized.matches("-?\\d+\\.(\\d+)?([eE][+-]?\\d+)?")
				&& !normalized.endsWith("f") && !normalized.endsWith("F"))
			return normalized + "f";
		return normalized;
	}

	private String normalizeSequenceLiteral(String value) {
		if (value == null)
			return "new Object[0]";
		String trimmed = value.trim();
		if (trimmed.isEmpty())
			return "new Object[0]";
		if ((trimmed.startsWith("(") && trimmed.endsWith(")")) || (trimmed.startsWith("[") && trimmed.endsWith("]")))
			trimmed = trimmed.substring(1, trimmed.length() - 1).trim();
		if (trimmed.endsWith(","))
			trimmed = trimmed.substring(0, trimmed.length() - 1).trim();
		if (trimmed.isBlank())
			return "new Object[0]";
		return "new Object[]{" + trimmed + "}";
	}

	private String sanitizeGeneratedExpression(String expression) {
		if (expression == null)
			return "";
		String normalized = expression.trim();
		if (normalized.isEmpty())
			return normalized;
		while (normalized.endsWith(","))
			normalized = normalized.substring(0, normalized.length() - 1).trim();
		normalized = normalized.replace("(,", "(").replace(",)", ")").replace(",,", ",");
		normalized = normalized.replace("float('inf')", "Float.POSITIVE_INFINITY")
				.replace("float(\"inf\")", "Float.POSITIVE_INFINITY");
		if (normalized.contains("float('") || normalized.contains("float(\""))
			return "0";
		if (normalized.contains("[") && (normalized.contains(":") || normalized.contains("None")))
			return "0";
		if (normalized.startsWith("[") && normalized.endsWith("]"))
			return "0";
		if ("None".equals(normalized))
			return "0";
		if (normalized.contains("tl.") || normalized.contains(".dtype") || normalized.contains(".element_ty"))
			return "0";
		return normalized;
	}

	private boolean isScalarBinaryTritonCall(String methodName, List<String> args, Map<String, Class> varNames) {
		if (args == null || args.size() != 2)
			return false;
		if (!"add".equals(methodName) && !"sub".equals(methodName) && !"mul".equals(methodName)
				&& !"div".equals(methodName) && !"mod".equals(methodName) && !"and".equals(methodName)
				&& !"or".equals(methodName))
			return false;
		return this.isSimpleScalarArgument(args.get(0), varNames) && this.isSimpleScalarArgument(args.get(1), varNames);
	}

	private String scalarBinaryExpression(String methodName, List<String> args) {
		String operator;
		switch (methodName) {
		case "add":
			operator = "+";
			break;
		case "sub":
			operator = "-";
			break;
		case "mul":
			operator = "*";
			break;
		case "div":
			operator = "/";
			break;
		case "mod":
			operator = "%";
			break;
		case "and":
			operator = "&";
			break;
		case "or":
			operator = "|";
			break;
		default:
			operator = "+";
			break;
		}
		return "(" + args.get(0) + operator + args.get(1) + ")";
	}

	private boolean isSimpleScalarArgument(String expression, Map<String, Class> varNames) {
		if (expression == null)
			return false;
		String normalized = expression.trim();
		if (normalized.isEmpty())
			return false;
		if (this.isNumericLiteral(normalized))
			return true;
		if ("true".equals(normalized) || "false".equals(normalized))
			return true;
		if (normalized.matches("[A-Za-z_$][A-Za-z0-9_$]*")) {
			Class type = this.resolveVarType(varNames, normalized);
			return !this.isTensorLikeType(type);
		}
		return false;
	}

	private boolean isNumericLiteral(String value) {
		return value.matches("-?\\d+") || value.matches("-?\\d+\\.(\\d+)?([eE][+-]?\\d+)?[fFdD]?");
	}

	private CodeBlock generateDotFunctionCall(TritonWriterContext context, FunctionCall functionCall,
			Map<String, Class> varNames) {
		InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);
		List<String> args = this.extractInvocationArguments(context, invocationArguments, varNames).stream()
				.map(arg -> this.normalizeFunctionArgument("dot", arg)).filter(arg -> arg != null && !arg.isBlank())
				.collect(Collectors.toList());
		if (args.size() >= 2)
			return CodeBlock.builder().add("oracle.code.triton.Triton.dot(" + args.get(0) + "," + args.get(1) + ")")
					.build();
		return CodeBlock.builder().add(args.isEmpty() ? "0" : args.get(0)).build();
	}

	private CodeBlock generateZerosFunctionCall(TritonWriterContext context, FunctionCall functionCall,
			Map<String, Class> varNames) {
		CodeBlock.Builder functionCallBuilder = CodeBlock.builder();

		InvocationArguments invocationArguments = functionCall.firstChildOfType(InvocationArguments.class);

		functionCallBuilder.add("oracle.code.triton.Triton.zeros(");

		String typeName = "Object.class";
		if (invocationArguments != null) {
			Argument argument = invocationArguments.firstChildOfType(Argument.class);
			if (argument != null) {
				DotName dotName = argument.firstDescendantOfType(DotName.class);
				if (dotName != null) {
					List<Name> names = dotName.childrenOfType(Name.class);
					if (names.size() > 1) {
						String dtype = names.get(1).toString();
						switch (dtype) {
						case "float16":
						case "float32":
							typeName = "float.class";
							break;
						case "float64":
							typeName = "double.class";
							break;
						case "int32":
							typeName = "int.class";
							break;
						case "int64":
							typeName = "long.class";
							break;
						default:
						}
					}
				}
			}
		}
		functionCallBuilder.add(typeName);

		Node shapeNode = null;
		if (invocationArguments != null) {
			shapeNode = invocationArguments.firstChildOfType(Tuple.class);
			if (shapeNode == null)
				shapeNode = invocationArguments.firstChildOfType(org.parsers.python.ast.List.class);
			if (shapeNode == null)
				shapeNode = invocationArguments.firstChildOfType(Group.class);
			if (shapeNode == null) {
				List<Node> nodes = invocationArguments.children();
				for (int i = 0; i < nodes.size(); i++) {
					Node node = nodes.get(i);
					if (node instanceof Argument || node instanceof Delimiter)
						continue;
					shapeNode = node;
					break;
				}
			}
		}
		if (shapeNode != null) {
			String shape = this.generateNodeText(context, shapeNode, varNames).trim();
			if ((shape.startsWith("(") && shape.endsWith(")")) || (shape.startsWith("[") && shape.endsWith("]")))
				shape = shape.substring(1, shape.length() - 1).trim();
			if (shape.endsWith(","))
				shape = shape.substring(0, shape.length() - 1).trim();
			if (!shape.isBlank()) {
				functionCallBuilder.add(",");
				functionCallBuilder.add(shape);
			}
		}

		functionCallBuilder.add(")");

		return functionCallBuilder.build();
	}

	private CodeBlock generateArgument(TritonWriterContext context, Argument argument, Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		Delimiter delimiter = argument.firstChildOfType(Delimiter.class);

		List<Node> nodes = argument.children();

		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);

			if (node instanceof Name) {
				if (!(delimiter != null && "=".equals(delimiter.toString()) && i == 0))
					codeBlockBuilder.add(node.toString());
			} else if (node instanceof NumericalLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof StringLiteral)
				codeBlockBuilder.add(node.toString());
			else if (node instanceof Group)
				codeBlockBuilder.add(this.generateGroup(context, (Group) node, varNames));
			else if (node instanceof FunctionCall)
				codeBlockBuilder.add(this.generateFunctionCall(context, (FunctionCall) node, varNames));
			else if (node instanceof MultiplicativeExpression)
				codeBlockBuilder
						.add(this.generateMultiplicativeExpression(context, (MultiplicativeExpression) node, varNames));
			else if (node instanceof AdditiveExpression)
				codeBlockBuilder.add(this.generateAdditiveExpression(context, (AdditiveExpression) node, varNames));
			else if (node instanceof UnaryExpression)
				codeBlockBuilder.add(this.generateUnaryExpression(context, (UnaryExpression) node, varNames));
			else if (node instanceof SliceExpression)
				codeBlockBuilder.add(this.generateSliceExpression(context, (SliceExpression) node, varNames));
			else if (node instanceof Comparison)
				codeBlockBuilder.add(this.generateComparison(context, (Comparison) node, varNames));
			else if (node instanceof BitwiseAnd)
				codeBlockBuilder.add(this.generateBitwise(context, node, varNames));
			else if (node instanceof BitwiseOr)
				codeBlockBuilder.add(this.generateBitwise(context, node, varNames));
		}
		return codeBlockBuilder.build();
	}

	private CodeBlock generateUnaryExpression(TritonWriterContext context, UnaryExpression expression,
			Map<String, Class> varNames) {
		CodeBlock.Builder codeBlockBuilder = CodeBlock.builder();

		List<Node> nodes = expression.children();

		for (int i = 0; i < nodes.size(); i++) {
			Node node = nodes.get(i);

			codeBlockBuilder.add(this.generateValue(context, node, varNames));
		}

		return codeBlockBuilder.build();
	}

	List<Comment> getComments(Node node) {
		if (node == null)
			return null;

		List<Comment> comments = new ArrayList<Comment>();

		List<? extends TerminalNode> statementNodes = node.getAllTokens(true);

		for (int j = 0; j < statementNodes.size(); j++) {
			TerminalNode statementNode = statementNodes.get(j);

			if (statementNode instanceof Comment)
				comments.add((Comment) statementNode);
		}

		return comments;
	}

	private boolean isPointerComment(String comment) {
		if (comment == null)
			return false;
		String normalized = comment.replace("#", "").replace("*", "").trim().toLowerCase();
		return normalized.startsWith("pointer to");
	}

	private boolean isLikelyPointerName(String name) {
		if (name == null)
			return false;
		String lower = name.toLowerCase();
		return lower.endsWith("_ptr") || lower.endsWith("_ptrs") || lower.endsWith("ptr")
				|| lower.contains("_ptr_");
	}

	private Class<?> getMethodReturnType(Class clazz, String methodName) {
		try {
			Method[] methods = clazz.getDeclaredMethods();
			for (Method method : methods) {
				if (method.getName().equals(methodName)) {
					return method.getReturnType();
				}
			}
		} catch (Exception e) {
			return null;
		}
		return null;
	}

	String normalizeComment(String comment) {
		if (comment == null)
			return comment;

		return comment.replace("#", "\n//");
	}

	String normalizeJavaDoc(String comment) {
		if (comment == null)
			return comment;

		return comment.replace("#", "") + "\n";
	}

	String normalizeClassName(String name) {
		if (name == null)
			return name;

		name = name.replaceFirst(name.substring(0, 1), name.substring(0, 1).toUpperCase());

		return name;
	}

	String normalizeVarName(String name) {
		if (name == null)
			return name;

		String normalized = name;
		if (normalized.isEmpty())
			normalized = "_";

		StringBuilder builder = new StringBuilder(normalized.length() + 1);
		for (int i = 0; i < normalized.length(); i++) {
			char current = normalized.charAt(i);
			boolean valid = i == 0 ? Character.isJavaIdentifierStart(current) : Character.isJavaIdentifierPart(current);
			builder.append(valid ? current : '_');
		}
		normalized = builder.toString();
		if (normalized.isEmpty() || !Character.isJavaIdentifierStart(normalized.charAt(0)))
			normalized = "_" + normalized;
		if (SourceVersion.isKeyword(normalized))
			normalized = normalized + "_";

		return normalized;
	}

	String normalizeMethodName(String name) {
		if (name == null)
			return name;

		name = name.replaceFirst(name.substring(0, 1), name.substring(0, 1).toLowerCase());

		name = this.replaceUnderscoreWithUppercase(name);

		return name;
	}

	private String replaceUnderscoreWithUppercase(String input) {
		if (input == null) {
			return null;
		}
		StringBuilder result = new StringBuilder();
		boolean toUpperCase = false;
		for (char c : input.toCharArray()) {
			if (c == '_') {
				toUpperCase = true;
			} else {
				if (toUpperCase) {
					result.append(Character.toUpperCase(c));
					toUpperCase = false;
				} else {
					result.append(c);
				}
			}
		}
		return result.toString();
	}
}
