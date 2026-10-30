package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// History built with the Anthropic Python SDK's model_dump() carries
// "citations": null on every text block.
const anthropicModelDumpHistoryRequest = `{
 "model":"chat","max_tokens":64,"messages":[
  {"role":"user","content":"read a"},
  {"role":"assistant","content":[{"citations":null,"text":"let me","type":"text"},
   {"id":"toolu_1","caller":null,"input":{"p":"a"},"name":"read","type":"tool_use","toolset_name":null}]},
  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}
 ]
}`

func TestAnthropicModelDumpHistoryDispatchesToChat(t *testing.T) {
	router := routingTestRouter("chat")
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(anthropicModelDumpHistoryRequest))
	if err != nil {
		t.Fatalf("ingress rejected model_dump history: %v", err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch rejected model_dump history: %v", err)
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		t.Fatalf("target = %q, want %q", dispatch.targetFormat, llmprotocol.OpenAIChatV1)
	}
	if _, err := router.encodeDispatchRequest(ctx); err != nil {
		t.Fatalf("encode rejected model_dump history: %v", err)
	}
}
