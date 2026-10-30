package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The Anthropic Python SDK's TextBlock.model_dump() emits "citations": null,
// so agents that store history that way resend it on every turn.
func TestAnthropicNullCitationsAreAbsent(t *testing.T) {
	engine := NewBuiltinEngine()
	request := `{"model":"m","max_tokens":64,"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"hi","citations":null}]},` +
		`{"role":"assistant","content":[{"citations":null,"text":"let me","type":"text"},` +
		`{"id":"toolu_1","caller":null,"input":{"p":"a"},"name":"read","type":"tool_use","toolset_name":null}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"ok","citations":null}]}]}]}`
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		if _, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, []byte(request), nil); err != nil {
			t.Fatalf("request with null citations -> %s: %v", target, err)
		}
	}

	response := `{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"hello","citations":null}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}}`
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, []byte(response), nil)
		if err != nil {
			t.Fatalf("response with null citations -> %s: %v", target, err)
		}
		if len(translated.Response.Output) != 1 || translated.Response.Output[0].Content[0].Text != "hello" {
			t.Fatalf("response text lost -> %s: %+v", target, translated.Response.Output)
		}
	}
}

func TestAnthropicNonNullCitationsStillRejected(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, citations := range []string{`[{"type":"char_location","cited_text":"h","document_index":0,"start_char_index":0,"end_char_index":1}]`} {
		request := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":[{"type":"text","text":"x","citations":` + citations + `}]},` +
			`{"role":"user","content":"y"}]}`
		_, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(request), nil)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
	}
}
