package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const redactedReasoningHistory = `{"model":"m","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[` +
	`{"role":"user","content":"hi"},` +
	`{"role":"assistant","content":[{"type":"redacted_thinking","data":"EmwKAhgB/opaque+payload=="},{"type":"text","text":"x"}]},` +
	`{"role":"user","content":"y"}]}`

func TestAnthropicRedactedThinkingRequestReplaysUnchanged(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(redactedReasoningHistory), nil)
	if err != nil {
		t.Fatalf("Messages to Messages rejected redacted_thinking: %v", err)
	}
	if !strings.Contains(string(translated.Body), `{"type":"redacted_thinking","data":"EmwKAhgB/opaque+payload=="}`) {
		t.Fatalf("redacted_thinking was not replayed byte-for-byte: %s", translated.Body)
	}
}

func TestAnthropicRedactedThinkingRequestRejectedCrossFormat(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, []byte(redactedReasoningHistory), nil)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
	}
}

func TestAnthropicRedactedThinkingResponseReplaysUnchanged(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"redacted_thinking","data":"abc"},{"type":"text","text":"hello"}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}}`
	engine := NewBuiltinEngine()
	translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(body), nil)
	if err != nil {
		t.Fatalf("Messages to Messages rejected a redacted_thinking response: %v", err)
	}
	var decoded struct {
		Content []map[string]any `json:"content"`
	}
	if err = json.Unmarshal(translated.Body, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Content) != 2 || decoded.Content[0]["type"] != "redacted_thinking" || decoded.Content[0]["data"] != "abc" {
		t.Fatalf("response content = %v", decoded.Content)
	}
	_, err = engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "")
}

func TestAnthropicRedactedThinkingRequiresData(t *testing.T) {
	body := strings.Replace(redactedReasoningHistory, `"data":"EmwKAhgB/opaque+payload=="`, `"data":""`, 1)
	_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "")
}
