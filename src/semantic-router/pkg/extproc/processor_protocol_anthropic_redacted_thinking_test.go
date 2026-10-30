package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// The second turn of an extended-thinking conversation in which Claude
// redacted part of its reasoning; the client must send the block back as is.
const anthropicRedactedThinkingHistory = `{"model":"test-model","max_tokens":2048,` +
	`"thinking":{"type":"enabled","budget_tokens":1024},"messages":[` +
	`{"role":"user","content":"hi"},` +
	`{"role":"assistant","content":[{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3p"},{"type":"text","text":"x"}]},` +
	`{"role":"user","content":"y"}]}`

func TestAnthropicRedactedThinkingDispatchesToAnthropicBackend(t *testing.T) {
	router := routingTestRouter("test-model")
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(anthropicRedactedThinkingHistory))
	if err != nil {
		t.Fatalf("ingress rejected redacted_thinking: %v", err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "test-model", "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch rejected redacted_thinking: %v", err)
	}
	if dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 {
		t.Fatalf("target = %q, want %q", dispatch.targetFormat, llmprotocol.AnthropicMessagesV1)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode rejected redacted_thinking: %v", err)
	}
	if !strings.Contains(string(body), `"data":"EmwKAhgBEgy3va3p"`) || !strings.Contains(string(body), `"redacted_thinking"`) {
		t.Fatalf("backend body lost the redacted block: %s", body)
	}
}
