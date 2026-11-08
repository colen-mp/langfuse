import {
  ReferenceNode,
  type KyselyPlugin,
  type PluginTransformQueryArgs,
  type PluginTransformResultArgs,
  type RootOperationNode,
} from "kysely";

import { QueryCompileError } from "./errors";
import { walkNodes } from "./nodes";

/**
 * Rejects every three-part column reference (`a.b.c`, from `sql.ref` or a
 * hand-built node). The passes after it read a reference's middle part as
 * its relation, while ClickHouse may read `e.col.field` as subcolumn `field`
 * of column `e.col`, so routing and tenancy would decide for the wrong
 * relation. Runs first in every compile. FROM/JOIN table names are not
 * references; `ctes.ts` rules on those.
 */
export class SchemaReferencePlugin implements KyselyPlugin {
  transformQuery(args: PluginTransformQueryArgs): RootOperationNode {
    walkNodes(args.node, (node) => {
      if (ReferenceNode.is(node) && node.table?.table.schema) {
        const { schema, identifier } = node.table.table;
        throw new QueryCompileError(
          `"${schema.name}.${identifier.name}" qualifies a column with two parts; the compile passes read only one. For a field of an Array(Tuple(…)) column use tupleArrayField(eb.ref("${schema.name}.${identifier.name}"), "<field>").`,
        );
      }
    });
    return args.node;
  }

  async transformResult(args: PluginTransformResultArgs) {
    return args.result;
  }
}
